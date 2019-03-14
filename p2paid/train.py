import os
import tempfile
from datetime import timedelta
from time import time

import pandas as pd
import sys

from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sqlalchemy import MetaData
import numpy as np
from sqlalchemy.dialects import mysql
from sqlalchemy.orm import aliased

sys.path.extend(['/home/andrei/Python/sqlalchemy-lab/p2paid'])
from utils import get_orders_info, evaluate_by_chunks, plot_qq_plot, plot_z_score_distrib, plot_z_score_abs_distrib, \
    get_r2_coeff

import sqlalchemy
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.expression import Insert

@compiles(Insert)
def append_string(insert, compiler, **kw):
    s = compiler.visit_insert(insert, **kw)
    if 'append_string' in insert.kwargs:
        return s + " " + insert.kwargs['append_string']
    return s

pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 2000)
pd.set_option('display.line_width', 2000)

test_size = 0.2
FILL_NA = timedelta(days=30)
# FEATURES = ['messages_count', 'edits_count', 'writer_approved_count', 'canceled_bids_count', 'paid_order_count', 'chat_count', 'dt_order_placed_secs']
#FEATURES = ['messages_count', 'edits_count', 'writer_approved_count', 'canceled_bids_count', 'chat_count', 'dt_order_placed_secs']

FEATURES = ['messages_count', 'edits_count', 'writer_approved_count', 'canceled_bids_count', 'chat_count',
            'dt_last_bid_cancel_secs', 'dt_last_chat_secs', 'dt_last_edit_secs', 'dt_last_message_secs', 'dt_last_writer_approved_secs',
            'dt_order_placed_secs', 'already_predicted_p2p']

# DT_FEATURES = ['dt_last_bid_cancel', 'dt_last_chat', 'dt_last_edit', 'dt_last_message', 'dt_last_writer_approved', 'dt_last_paid_order']
DT_FEATURES = ['dt_last_bid_cancel', 'dt_last_chat', 'dt_last_edit', 'dt_last_message', 'dt_last_writer_approved']
# COUNT_FEATURES = ['canceled_bids_count', 'chat_count', 'edits_count', 'messages_count', 'paid_order_count', 'writer_approved_count']
COUNT_FEATURES = ['canceled_bids_count', 'chat_count', 'edits_count', 'messages_count', 'writer_approved_count']

model_params = dict(
    hidden_layer_sizes=(32, 16, 16, 8, 4),
    max_iter=1000, verbose=20, tol=1e-5,
    activation='tanh', solver='adam'
)


def prepare_df(features_df: pd.DataFrame) -> pd.DataFrame:
    for k in DT_FEATURES + COUNT_FEATURES:
        assert k in features_df.columns, "{} field not in features columns {}".format(k, ','.join(features_df.columns))

    features_df[DT_FEATURES] = features_df[DT_FEATURES].fillna(FILL_NA)

    if 'order_created_p2p' in features_df.columns:
        features_df.order_created_p2p = features_df.order_created_p2p.fillna(0)

    features_df[['{}_secs'.format(key) for key in DT_FEATURES]] = features_df[DT_FEATURES].apply(lambda x: x.dt.total_seconds())
    features_df[['dt_order_placed_secs']] = features_df[['dt_order_placed']].apply(lambda x: x.dt.total_seconds())

    ################### observed p2p proba ##############################
    observed_p2p_df = features_df.groupby('dt_order_placed').is_paid_order.mean().to_frame('observed_p2p')
    features_df = features_df.join(observed_p2p_df.observed_p2p, on='dt_order_placed')

    features_df[COUNT_FEATURES] = features_df[COUNT_FEATURES].fillna(0)
    return features_df

def get_additional_columns(features_df):
    ids = features_df.index.unique()
    orders_df = get_orders_info(ids)

    if 'is_paid_order' not in features_df.columns:
        features_df = features_df.join(orders_df.is_paid_order)

    if 'order_created_p2p' not in features_df.columns:
        features_df = features_df.join(orders_df.place2paid_proba)
        features_df = features_df.rename(columns={"place2paid_proba": "order_created_p2p"})

    return features_df

def get_train_test_features(features_df: pd.DataFrame, orders_df: pd.DataFrame=None) -> tuple:
    features_df = features_df.reset_index()
    features_df = features_df.sample(frac=1).set_index('order_id')

    ids = features_df.index.unique()

    features_df = get_additional_columns(features_df)
    # if orders_df is None:
    #     orders_df = get_orders_info(ids)
    #
    #     if 'is_paid_order' not in features_df.columns:
    #         features_df = features_df.join(orders_df.is_paid_order)
    #
    #     if 'order_created_p2p' not in features_df.columns:
    #         features_df = features_df.join(orders_df.place2paid_proba)
    #         features_df = features_df.rename(columns={"place2paid_proba": "order_created_p2p"})

    features_df = prepare_df(features_df)
    # original_df_mean = orders_df.is_paid_order.mean()
    # features_df_mean = features_df.groupby('order_id').is_paid_order.apply(lambda x: x.max()).mean()
    # assert(original_df_mean == features_df_mean), "original_df_mean AND features_df_mean does not equal"

    # add additional feature
    # features_df = features_df.reset_index().set_index(['order_id', 'dt_order_placed'])
    # features_df['dt_min_event'] = features_df[DT_FEATURES].min(axis=1)
    # features_df['dt_min_event_secs'] = features_df.dt_min_event.apply(lambda x: x.total_seconds())
    # features_df = features_df.reset_index().set_index('order_id')

    ids_train, ids_test = train_test_split(ids, test_size=test_size)

    df_train = features_df[features_df.index.isin(ids_train)]
    df_test = features_df[features_df.index.isin(ids_test)]
    return df_train, df_test


def train_model(df_train: pd.DataFrame):
    mlp = MLPClassifier(**model_params)

    piple = Pipeline([
        ('scale', StandardScaler()),
        ('mlp', mlp),
    ])

    piple.fit(df_train[FEATURES], df_train.is_paid_order)  # can be passed generator

    return piple


def get_p2p_mse_r2(df: pd.DataFrame) -> tuple:
    """
    1. plot: p2p_proba=F(dt) vs p2p_obs=F(dt)
    2. plot: p2p_proba vs p2p_obs for diff dt_placed_order / calc MSE & R^2 - GLOBAL assessment metric
    :param df:
    :return tuple: (DataFrame, MSE error, R^2)
    """
    compare_p2p = df.groupby('dt_order_placed')[['is_paid_order', 'p2p_proba']].mean()
    compare_p2p.index = compare_p2p.index.total_seconds() / 60
    y_true, y_pred = compare_p2p.values[:, 0], compare_p2p.values[:, 1]
    mse_train = mean_squared_error(y_true, y_pred)
    r2_train = r2_score(y_true, y_pred)
    return compare_p2p, mse_train, r2_train

def plot_hist_p2p_proba(df):
    """
    plot: histogram: p2p_proba for different dt_placed_order / try to make a GIF
    :param df:
    :return None:
    """

# import shutil
# with open("/home/andrei/Python/sqlalchemy-lab/model_1/test_from_class.png", 'wb') as fdst:
#     shutil.copyfileobj(tmpfile, fdst)
#
# tmpfile.seek(0)
# im = imageio.imread(tmpfile)
# imageio.imwrite("/home/andrei/Python/sqlalchemy-lab/model_1/imageio.png", im)


class Plt2GIF:
    def __init__(self):
        self.images = []

    def add_image(self, fig):
        import imageio
        with tempfile.TemporaryFile(suffix=".png") as tmpfile:
            fig.savefig(tmpfile)
            tmpfile.seek(0)
            self.images.append(imageio.imread(tmpfile))

    def make_gif(self, output_file, duration=1.2):
        import imageio
        imageio.mimsave(output_file, self.images, duration=duration)

    def save_frames(self, output_file):
        import imageio

        output_file = output_file.replace('.', '{}.')

        for i, im in enumerate(self.images):
            imageio.imwrite(output_file.format(i), im)

class ModelSaver:
    def __init__(self, folder=None):
        self.folder = self.create_folder_if_not_exists(folder)

    def create_folder_if_not_exists(self, folder: str) -> str:
        if not folder:
            folder = 'models/model_1'
            n = 2
            while os.path.exists(folder):
                folder = 'models/model_{}'.format(n)
                n += 1

        if not os.path.exists(folder):
            os.makedirs(folder)

        return folder

    def save_graph(self, fig, name):
        fig.savefig('{}/{}'.format(self.folder, name), format="png")

    def get_folder(self):
        return self.folder

    def save_model(self, model):
        filename = 'model.sav'
        joblib.dump(model, '{}/{}'.format(self.folder, filename))

    @staticmethod
    def load_model(folder):
        filename = 'model.sav'
        return joblib.load('{}/{}'.format(folder, filename))

    def log(self, *args, end:str="\n") -> None:
        assert isinstance(end, str), 'end delimiter should be a string'
        with open('{}/model.info'.format(self.folder), 'a+') as f:
            f.write(" ".join([str(i) for i in args]) + end)


def evaluate_model(model: Pipeline, df_train: pd.DataFrame, df_test: pd.DataFrame):
    """
    Evaluate Model
    :param model:
    :param df_train:
    :param df_test:
    :return: None
    """

    saver = ModelSaver()
    print('FEATURES:', FEATURES)
    saver.log('FEATURES:', FEATURES)
    print('Model params:', model_params)
    saver.log('Model params:', model_params)

    y_pred_proba = model.predict_proba(df_train[FEATURES])
    print('Train Predicted', y_pred_proba.mean(0))
    saver.log('Train Predicted', y_pred_proba.mean(0))

    y_pred_proba = model.predict_proba(df_test[FEATURES])
    print('Test Predicted', y_pred_proba.mean(0))
    saver.log('Test Predicted', y_pred_proba.mean(0))

    df_train['p2p_proba'] = model.predict_proba(df_train[FEATURES])[:, 1]
    df_test['p2p_proba'] = model.predict_proba(df_test[FEATURES])[:, 1]

    p2pdf, mse_train_mean, r2_train_mean = get_p2p_mse_r2(df_train)
    p2pdf.plot()
    plt.title("plot: Train p2p_proba=F(dt) vs p2p_obs=F(dt)")
    saver.save_graph(plt, '1_train_p2p_proba_&_obs.png')
    # plt.show()
    plt.cla()

    p2pdf, mse_test_mean, r2_test_mean = get_p2p_mse_r2(df_test)
    p2pdf.plot()
    plt.title("plot: Test p2p_proba=F(dt) vs p2p_obs=F(dt)")
    saver.save_graph(plt, '1_test_p2p_proba_&_obs.png')
    # plt.show()
    plt.cla()

    print('train: MSE', mse_train_mean, 'R^2', r2_train_mean)
    print('test: MSE ', mse_test_mean, 'R^2', r2_test_mean)
    saver.log('train: MSE', mse_train_mean, 'R^2', r2_train_mean)
    saver.log('test: MSE ', mse_test_mean, 'R^2', r2_test_mean)

    build_p2p_hist_and_save_to_gif(df_train, gif_name="{}/2_hist_p2p_proba.gif".format(saver.get_folder()))
    plt.cla()

    plot_std(df_train)

    r2, z, z_abs = evaluate_model_by_times(df_train, saver)
    print(f"Train: R2: {r2:.5f}, z: {z:.5f}, z_abs: {z_abs:.5f}")
    saver.log(f"Train: R2: {r2:.5f}, z: {z:.5f}, z_abs: {z_abs:.5f}")

    r2_test, z_test, z_abs_test = evaluate_model_by_times(df_test, saver, is_test=True)
    print(f"Test: R2: {r2_test:.5f}, z: {z_test:.5f}, z_abs: {z_abs_test:.5f}")
    saver.log(f"Test: R2: {r2_test:.5f}, z: {z_test:.5f}, z_abs: {z_abs_test:.5f}")

    saver.save_model(model)

    return dict(
        r2_train_batch_mean=float(f"{r2:.5f}"),
        r2_train_mean=float(f"{r2_train_mean:.5f}"),
        mse_train_mean=float(f"{mse_train_mean:.5f}"),

        r2_test_batch_mean=float(f"{r2_test:.5f}"),
        r2_test_mean=float(f"{r2_test_mean:.5f}"),
        mse_test_mean=float(f"{mse_test_mean:.5f}"),

        z=float(f"{z:.5f}"),
        z_abs=float(f"{z_abs:.5f}"),

        z_test=float(f"{z_test:.5f}"),
        z_abs_test=float(f"{z_abs_test:.5f}")
    )



def evaluate_model_by_times(df, saver=None, is_test=False, save_frames=False) -> tuple:
    # df = df_train
    # df = df_test
    df['is_first_client_order'] = 1
    df['is_paid'] = df.is_paid_order

    times = df.groupby('dt_order_placed').dt_order_placed.max().to_dict().values()

    plt_2_gif = Plt2GIF()
    z_score = 0
    z_score_abs = 0
    r2_score_total = 0

    min_p2p_obs, min_p2p = df[['observed_p2p', 'p2p_proba']].min().values
    max_p2p_obs, max_p2p = df[['observed_p2p', 'p2p_proba']].max().values
    boundaries = (min(min_p2p_obs, min_p2p), max(max_p2p_obs, max_p2p))

    chunk_size = min(df.groupby('dt_order_placed').is_paid.count().min(), 1000)

    for t in times:
        df_test = df[df.dt_order_placed == t]
        df_eval = evaluate_by_chunks(df_test, chunk_size)

        plt.cla()
        axes = plt.gca()
        axes.set_xlim([0, max_p2p_obs])
        axes.set_ylim([0, max_p2p])
        fig, ax = plot_qq_plot2(df_eval, boundaries=boundaries)
        ax.set_title('place2paid: observed rate vs predicted proba for time {}'.format(t))
        plt_2_gif.add_image(fig)
        r2 = get_r2_coeff(df_eval)
        r2_score_total += 0 if np.isnan(r2) else r2
        z_score += df_eval.z_score.mean()
        z_score_abs += df_eval.z_score_abs.mean()

    if not is_test:
        plt_2_gif.make_gif('{}/3_train_qq_plot.gif'.format(saver.get_folder()))

        if save_frames:
            plt_2_gif.save_frames('{}/3_train_qq_frames.png'.format(saver.get_folder()))
    else:
        plt_2_gif.make_gif('{}/3_test_qq_plot.gif'.format(saver.get_folder()))

        if save_frames:
            plt_2_gif.save_frames('{}/3_test_qq_frames.png'.format(saver.get_folder()))

    return r2_score_total / len(times), z_score / len(times), z_score_abs / len(times),


def test_fig():
    t = np.arange(0.0, 2.0, 0.01)
    s = 1 + np.sin(2 * np.pi * t)

    fig, ax = plt.subplots()
    ax.plot(t, s)

    ax.set(xlabel='time (s)', ylabel='voltage (mV)',
           title='About as simple as it gets, folks')
    dir(ax)
    ax.grid()

    # fig.savefig("test.png")
    # plt.show()


def plot_qq_plot2(df_eval, CHUNK_SIZE=1000, boundaries=None):
    """plot p2p observed rate vs predicted proba Q-Q plot"""

    if not boundaries:
        start = df_eval[['p2p_obs', 'p2p_proba']].values.min()
        end = df_eval[['p2p_obs', 'p2p_proba']].values.max()
    else:
        start, end = boundaries

    p = np.linspace(start, end, 100)
    std_err = np.sqrt(p * (1 - p) / CHUNK_SIZE)
    conf_interv = 2 * std_err

    # plt.figure()
    fig, ax = plt.subplots()
    ax.plot(p, p, 'b--')
    ax.plot(df_eval.p2p_obs, df_eval.p2p_proba, 'ro', color='red')
    ax.fill_between(p, p + conf_interv, p - conf_interv, facecolor='yellow')
    # ax.set_title('place2paid: observed rate vs predicted proba')
    ax.set_xlabel('observed p2p rate')
    ax.set_ylabel('predicted p2p proba')
    return fig, ax

def plot_std(df):
    """
    4. plot: std(p2p_proba) for diff dt_placed_order
    :return: None
    """
    p2p_std_df = df.groupby('dt_order_placed').p2p_proba.std()
    p2p_std_df.index = p2p_std_df.index.total_seconds() / 60
    p2p_std_df.plot()


def build_p2p_hist_and_save_to_gif(df: pd.DataFrame, gif_name='hist_p2p_proba.gif', duration=1.2):
    """
    3. plot: histogram: p2p_proba for different dt_placed_order / try to make a GIF
    :param df:
    :param gif_name:
    :param duration:
    :return: None
    """
    plt_2_gif = Plt2GIF()
    max_orders = df[df.is_paid_order == 0].groupby('dt_order_placed').dt_order_placed.count().max()
    mins = df.groupby('dt_order_placed').dt_order_placed.apply(lambda x: x.iloc[0])
    for dt_min in mins:
        fig, ax = plt.subplots()
        check_df = df[df.dt_order_placed == dt_min]
        unpaid_ids = check_df.index.unique()

        paid_ids = df[(df.dt_order_placed < dt_min) & (df.is_paid_order == 1) & (~df.index.isin(unpaid_ids))].index.unique()
        paid_orders = df[df.index.isin(paid_ids)].groupby('order_id').p2p_proba.mean().to_frame()
        paid_orders.p2p_proba = 1
        check_df = check_df.append(paid_orders)
        check_df.p2p_proba.hist(bins=30)
        ax.set_title('Time: {}, paid orders: {}, unpaid: {}'.format(dt_min, len(paid_orders), len(unpaid_ids)))
        ax.set_xlim([0, 1])
        ax.set_ylim([0, max_orders])
        plt_2_gif.add_image(fig)
        # plt.show()
        plt.clf()

    plt_2_gif.make_gif(gif_name, duration=duration)


def save_features(features_df:pd.DataFrame):
    conn = sqlalchemy.create_engine('mysql+pymysql://root@localhost/edusson_data_science')

    files = ['features_df_with_web_25000_orders_0_.pkl']
    files += ['features_df_with_web_25000_orders_25000.pkl']
    files += ['features_df_with_web_25000_orders_50000.pkl']
    files += ['features_df_with_web_25000_orders_75000.pkl']
    files += ['features_df_with_web_25000_orders_100000.pkl']
    files += ['features_df_with_web_25000_orders_125000.pkl']
    files += ['features_df_with_web_25000_orders_150000.pkl']
    files += ['features_df_with_web_25000_orders_175000.pkl']
    files += ['features_df_with_web_25000_orders_200000.pkl']
    files += ['features_df_with_web_19533_orders_225000.pkl']

    meta = MetaData()
    meta.reflect(bind=conn)

    P2POrderFeatures = meta.tables['p2p_order_features']

    ins = P2POrderFeatures.insert(append_string = 'ON DUPLICATE KEY UPDATE order_id=order_id, dt_order_placed=dt_order_placed')
    str(ins.compile())

    for file in files:
        print('load file', file)
        df = pd.read_pickle('/home/andrei/Python/sqlalchemy-lab/p2paid/{}'.format(file))
        print('df count', len(df))
        stime = time()

        for i in range(0, len(df), 100000):

            t_df = df[i:i+100000].reset_index()

            t_df[COUNT_FEATURES] = t_df[COUNT_FEATURES].fillna(0)

            for col in DT_FEATURES:
                t_df[col] = t_df[col].apply(lambda x: None if pd.isna(x) else x.total_seconds())

            t_df.dt_order_placed = t_df.dt_order_placed.apply(lambda x: x.total_seconds())

            values = t_df[DT_FEATURES + COUNT_FEATURES + ['order_id', 'dt_order_placed']].to_dict('records')

            for row in values:
                for k,v in row.items():
                    if k in DT_FEATURES:
                        row[k] = float(v) if v and not np.isnan(v) else None

            result = conn.execute(ins, values)

            print('Done', end=' ')
            print(i, 'len', len(t_df), end=' ')
            print(str(result), end=' ')
            print('Time:', time()-stime)


from sqlalchemy import select, and_, func

def get_features_from_db(orders_ids=None, dt_placed_orders=None, orders_limit=None, return_query=False):
    conn = sqlalchemy.create_engine('mysql+pymysql://root@localhost/edusson_data_science')
    meta = MetaData()
    meta.reflect(bind=conn)

    P2POrderFeatures = meta.tables['p2p_order_features']

    query = select([P2POrderFeatures])

    if orders_ids and dt_placed_orders:
        query = query.where(and_(
            P2POrderFeatures.c.order_id.in_(orders_ids),
            P2POrderFeatures.c.dt_order_placed.in_(dt_placed_orders)
        ))
    else:
        if orders_ids:
            query = query.where(P2POrderFeatures.c.order_id.in_(orders_ids))

        if dt_placed_orders:
            query = query.where(P2POrderFeatures.c.dt_order_placed.in_(dt_placed_orders))

    if orders_limit:
        stmt = aliased(select([P2POrderFeatures.c.order_id])
                       .distinct(P2POrderFeatures.c.order_id)
                       .limit(orders_limit)
                       .order_by(func.random()))

        query = query.select_from(P2POrderFeatures.join(stmt, P2POrderFeatures.c.order_id == stmt.c.order_id))

    if return_query:
        return query

    return conn.execute(query).fetchall()


def get_dataframe_from_db(**kwargs) -> pd.DataFrame:
    query = get_features_from_db(return_query=True, **kwargs)
    conn = sqlalchemy.create_engine('mysql+pymysql://root@localhost/edusson_data_science')
    compiled_query = query.compile(dialect=mysql.dialect())
    df = pd.read_sql(sql=compiled_query, con=conn, params=compiled_query.params)
    df[DT_FEATURES] = df[DT_FEATURES].apply(lambda x: pd.to_timedelta(x, unit='s'))
    df.dt_order_placed = df.dt_order_placed.apply(lambda x: pd.to_timedelta(x, unit='s'))
    return df.drop(columns=['is_paid_order']).set_index('order_id')


def fill_non_exists_actions(df):
    for col in COUNT_FEATURES:
        if col not in df.columns:
            df[col] = 0

    for col in DT_FEATURES:
        if col not in df.columns:
            df[col] = np.NaN

    return df


def check_some_order():
    order_id = 1530281
    from concurent_utils import get_features
    from utils import get_actions

    info = get_orders_info([order_id]).reset_index()
    actions = get_actions(info)
    all_times = pd.concat((actions.dt_order_placed, pd.Series([timedelta(seconds=stime) for stime in times])))
    all_times = all_times.sort_values(0).drop_duplicates().reset_index(drop=True)
    order_features = pd.concat([get_features(actions, dt_order_placed) for k, dt_order_placed in all_times.iteritems()])


def build_graph_actions_by_order(order_id, model, times, slice_by_mins=None):
    from concurent_utils import get_features
    from utils import get_actions

    print('order_id', order_id)

    info = get_orders_info([order_id]).reset_index()
    actions = get_actions(info)

    all_times = pd.concat((actions.dt_order_placed, pd.Series([timedelta(seconds=stime) for stime in times])))
    all_times = all_times.sort_values(0).drop_duplicates().reset_index(drop=True)
    order_features = pd.concat([get_features(actions, dt_order_placed) for k, dt_order_placed in all_times.iteritems()])

    order_features['order_id'] = order_id
    order_features = order_features.reset_index().set_index('order_id')
    order_features = fill_non_exists_actions(order_features)
    order_features = get_additional_columns(order_features)
    order_features = prepare_df(order_features)
    order_features = add_last_p2p_static_model_prediction(order_features)


    order_features = order_features.reset_index().set_index('dt_order_placed')
    dt_seconds = order_features.index
    p2p_vals = []

    prev_actions_df = pd.DataFrame(columns=COUNT_FEATURES, data=[[0] * len(COUNT_FEATURES)])
    prev_actions_df = pd.melt(prev_actions_df[COUNT_FEATURES], value_vars=COUNT_FEATURES)

    for dt_second in dt_seconds:
        check_df = order_features[order_features.index == dt_second]
        df_melted = pd.melt(check_df[COUNT_FEATURES], value_vars=COUNT_FEATURES)
        prediction = model.predict_proba(check_df[FEATURES])[:, 1][0]

        df_melted['diff_value'] = df_melted.value - prev_actions_df.value
        current_action = df_melted.sort_values(by='diff_value', ascending=False).iloc[0]
        current_action_name = ''

        if current_action.diff_value != 0:
            current_action_name = current_action.variable

        p2p_vals.append([prediction, current_action_name])
        prev_actions_df = df_melted

    p2p_vals = np.array(p2p_vals)

    order_features['p2p_proba'] = p2p_vals[:, 0].astype(np.float32)
    order_features['p2p_action'] = p2p_vals[:, 1]

    order_features.index = order_features.index.total_seconds() / 60

    if slice_by_mins:
        sliced = order_features[order_features.index < slice_by_mins]
    else:
        sliced = order_features.copy()

    sliced.p2p_proba.plot()

    plot_styles = {'canceled_bids_count': 'ro','chat_count': 'gv',
                        'edits_count': 'bo','messages_count': 'gv','writer_approved_count': 'r<'}

    plot_dict = {'canceled_bids_count': [],'chat_count': [],
                 'edits_count': [],'messages_count': [],'writer_approved_count': []}

    for kx, ky in sliced[sliced.p2p_action != ''][['p2p_proba', 'p2p_action']].iterrows():
        plot_dict[ky.p2p_action].append((kx, ky.p2p_proba))

    for k,v in plot_dict.items():
        if len(v):
            points = np.array(v)
            plt.plot(points[:, 0], points[:, 1], plot_styles[k], label=k)

    plt.legend()
    plt.title('{} order'.format(order_id))
    # plt.grid(color='r', linestyle='-', linewidth=0.5, axis='x', alpha=0.3, ydata=sliced.index.values)
    # plt.grid(color='r', linestyle='-', linewidth=0.5, alpha=0.3, xdata=sliced.index.values)

    plt.grid(color='r', which='both', linestyle='-', linewidth=0.5, axis='x', alpha=0.3)
    plt.show()
    plt.clf()

class Storage:
    cache = {}

    @classmethod
    def save(cls, key, val):
        cls.cache[key] = val

    @classmethod
    def load(cls, key):
        return cls.cache[key]

def load_all_p2p_df():
    from utils import make_tmp_p2p_mysql_table, get_p2p_proba_from_api_log, drop_tmp_p2p_mysql_table

    make_tmp_p2p_mysql_table()
    p2pdf = get_p2p_proba_from_api_log()
    drop_tmp_p2p_mysql_table()

    Storage.save('p2pdf', p2pdf)


def add_last_p2p_static_model_prediction(df, join=True):
    if 'already_predicted_p2p' in df.columns and join:
        return df

    # df = order_features.copy()

    p2pdf = Storage.load('p2pdf')

    orders_ids_with_p2p = p2pdf.index.unique()

    if join:
        order_with_p2p_df = df[df.index.isin(orders_ids_with_p2p)]
        order_with_p2p_df['already_predicted_p2p'] = p2pdf.to_frame().place2paid_proba
        # df['already_predicted_p2p'] = np.NaN
        # df['already_predicted_p2p'] = p2pdf.to_frame().place2paid_proba
        return order_with_p2p_df

    return p2pdf.to_frame().place2paid_proba





def plot_relationship_between_p2p_predicted_and_first_p2p_by_static_model(order_with_p2p_df, model):

    plt_2_gif = Plt2GIF()
    mins = order_with_p2p_df.groupby('dt_order_placed').dt_order_placed.apply(lambda x: x.iloc[0])
    order_with_p2p_df = get_additional_columns(order_with_p2p_df)
    order_with_p2p_df = prepare_df(order_with_p2p_df)
    for dt_min in mins:
        check_df = order_with_p2p_df[order_with_p2p_df.dt_order_placed == dt_min]
        check_df['p2p'] = model.predict_proba(check_df[FEATURES])[:, 1]

        check_df[['already_predicted_p2p', 'p2p']].plot.scatter(x='p2p', y='already_predicted_p2p', linewidths=0.001)
        plt.title(dt_min)
        plt.plot([0,1], [0,1])
        plt_2_gif.add_image(plt)
        plt.clf()

    plt_2_gif.make_gif('TEST_plot_relationship_between_p2p_predicted_and_first_p2p_by_static_model.gif')




if __name__ == '__main__':
    load_all_p2p_df()

    times = [0,21,31,40,49,59,68,78,89,100,111,124,137,152,167,184,200,218,236,256,278,301,326,352,380,409,440,473,510,549,591,637,688,743,802,870,943,1027,1124,1238,1372,1536,1740,2006,2370,2889,3733,5447,11782,81440]
    times = [t for i, t in enumerate(times) if i % 2 == 0]

    features_df = get_dataframe_from_db(orders_limit=1000000, dt_placed_orders=times)
    features_df = add_last_p2p_static_model_prediction(features_df)
    print('Mean Order Placed DT', features_df.groupby('order_id').dt_order_placed.count().mean())

    model = ModelSaver.load_model('/home/andrei/Python/sqlalchemy-lab/models/model_13')

    import random
    paid_ids = features_df[features_df.is_paid_order == 1].index.unique()
    build_graph_actions_by_order(random.choice(paid_ids), model, times)

    build_graph_actions_by_order(1530281, model, times, 25)


    order_with_p2p_df = add_last_p2p_static_model_prediction(features_df)


    df_train, df_test = get_train_test_features(order_with_p2p_df)
    model = train_model(df_train)
    res = evaluate_model(model, df_train, df_test)

    print('res', res)

    t0_df = order_with_p2p_df[order_with_p2p_df.dt_order_placed == timedelta(seconds=0)]
    t0_df.already_predicted_p2p.hist(bins=30)
    plt.show()

    t0_df = get_additional_columns(t0_df)
    t0_df = prepare_df(t0_df)
    t0_df['p2p'] = model.predict_proba(t0_df[FEATURES])[:, 1]
    t0_df.p2p.hist(bins=30)
    plt.show()


    plot_relationship_between_p2p_predicted_and_first_p2p_by_static_model(order_with_p2p_df, model)

    t0_df = order_with_p2p_df[order_with_p2p_df.dt_order_placed == timedelta(seconds=0)]
    # t0_df[['already_predicted_p2p', 'p2p']].reset_index().set_index('already_predicted_p2p').plot.scatter()
    t0_df[['already_predicted_p2p', 'p2p']].plot.scatter(x='p2p', y='already_predicted_p2p', linewidths=0.001, )
    plt.plot([0,1], [0,1])
    # np.arange(0,1,0.1)
    plt.show()