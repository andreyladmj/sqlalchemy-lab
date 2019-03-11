import os
import tempfile
from datetime import timedelta
from time import time

import pandas as pd
import sys

import sqlalchemy
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sqlalchemy import MetaData
import numpy as np

sys.path.extend(['/home/andrei/Python/sqlalchemy-lab/p2paid'])
from utils import get_orders_info, evaluate_by_chunks, plot_qq_plot, plot_z_score_distrib, plot_z_score_abs_distrib, \
    get_r2_coeff

pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 2000)
pd.set_option('display.line_width', 2000)

test_size = 0.2
FILL_NA = timedelta(days=30)
FEATURES = ['messages_count', 'edits_count', 'writer_approved_count', 'canceled_bids_count', 'paid_order_count', 'chat_count', 'dt_order_placed_secs']

DT_FEATURES = ['dt_last_bid_cancel', 'dt_last_chat', 'dt_last_edit', 'dt_last_message', 'dt_last_writer_approved', 'dt_last_paid_order']
COUNT_FEATURES = ['canceled_bids_count', 'chat_count', 'edits_count', 'messages_count', 'paid_order_count', 'writer_approved_count']

model_params = dict(
    hidden_layer_sizes=(32, 16, 16, 8, 4),
    max_iter=1000, verbose=10, tol=1e-5,
    activation='tanh', solver='adam'
)


def prepare_df(features_df: pd.DataFrame) -> pd.DataFrame:
    for k in DT_FEATURES + COUNT_FEATURES:
        assert k in features_df.columns, "{} field not in features columns {}".format(k, ','.join(features_df.columns))

    features_df[DT_FEATURES] = features_df[DT_FEATURES].fillna(FILL_NA)

    features_df[['{}_secs'.format(key) for key in DT_FEATURES]] = features_df[DT_FEATURES].apply(lambda x: x.dt.total_seconds())
    features_df[['dt_order_placed_secs']] = features_df[['dt_order_placed']].apply(lambda x: x.dt.total_seconds())

    ################### observed p2p proba ##############################
    observed_p2p_df = features_df.groupby('dt_order_placed').is_paid_order.mean().to_frame('observed_p2p')
    features_df = features_df.join(observed_p2p_df.observed_p2p, on='dt_order_placed')

    features_df[COUNT_FEATURES] = features_df[COUNT_FEATURES].fillna(0)
    return features_df


def get_train_test_features(features_df: pd.DataFrame, orders_df: pd.DataFrame=None) -> tuple:
    features_df = features_df.reset_index().set_index('order_id')

    ids = features_df.index.unique()

    if orders_df is None:
        orders_df = get_orders_info(ids)

    features_df = features_df.join(orders_df.is_paid_order)

    features_df = prepare_df(features_df)
    original_df_mean = orders_df.is_paid_order.mean()
    features_df_mean = features_df.groupby('order_id').is_paid_order.apply(lambda x: x.max()).mean()
    assert(original_df_mean == features_df_mean), "original_df_mean AND features_df_mean does not equal"

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

class Plt2GIF:
    def __init__(self):
        self.images = []

    def add_image(self, fig):
        import imageio
        with tempfile.TemporaryFile(suffix=".png") as tmpfile:
            fig.savefig(tmpfile, format="png")
            tmpfile.seek(0)
            self.images.append(imageio.imread(tmpfile))

    def make_gif(self, output_file, duration=1.2):
        import imageio
        imageio.mimsave(output_file, self.images, duration=duration)

class ModelSaver:
    def __init__(self, folder=None):
        self.folder = self.create_folder_if_not_exists(folder)

    def create_folder_if_not_exists(self, folder: str) -> str:
        if not folder:
            folder = 'model_1'
            n = 2
            while os.path.exists(folder):
                folder = 'model_{}'.format(n)
                n += 1

        if not os.path.exists(folder):
            os.makedirs(folder)

        return folder

    def save_graph(self, fig, name):
        fig.savefig('{}/{}'.format(self.folder, name), format="png")

    def get_folder(self):
        return self.folder

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
    print('FEATURES', FEATURES)
    saver.log('FEATURES', FEATURES)

    y_pred_proba = model.predict_proba(df_train[FEATURES])
    print('Train Predicted', y_pred_proba.mean(0))
    saver.log('Train Predicted', y_pred_proba.mean(0))

    y_pred_proba = model.predict_proba(df_test[FEATURES])
    print('Test Predicted', y_pred_proba.mean(0))
    saver.log('Test Predicted', y_pred_proba.mean(0))

    df_train['p2p_proba'] = model.predict_proba(df_train[FEATURES])[:, 1]
    df_test['p2p_proba'] = model.predict_proba(df_test[FEATURES])[:, 1]

    p2pdf, mse_train, r2_train = get_p2p_mse_r2(df_train)
    p2pdf.plot()
    plt.title("plot: Train p2p_proba=F(dt) vs p2p_obs=F(dt)")
    saver.save_graph(plt, '1_train_p2p_proba_&_obs.png')
    # plt.show()
    plt.cla()

    p2pdf, mse_test, r2_test = get_p2p_mse_r2(df_test)
    p2pdf.plot()
    plt.title("plot: Test p2p_proba=F(dt) vs p2p_obs=F(dt)")
    saver.save_graph(plt, '1_test_p2p_proba_&_obs.png')
    # plt.show()
    plt.cla()

    print('train: MSE', mse_train, 'R^2', r2_train)
    print('test: MSE ', mse_test, 'R^2', r2_test)
    saver.log('train: MSE', mse_train, 'R^2', r2_train)
    saver.log('test: MSE ', mse_test, 'R^2', r2_test)

    build_p2p_hist_and_save_to_gif(df_train, gif_name="{}/2_hist_p2p_proba.gif".format(saver.get_folder()))
    plt.cla()

    plot_std(df_train)

    r2, z, z_abs = evaluate_model_by_times(df_train, saver)
    print(f"Train: R2: {r2:.5f}, z: {z:.5f}, z_abs: {z_abs:.5f}")
    saver.log(f"Train: R2: {r2:.5f}, z: {z:.5f}, z_abs: {z_abs:.5f}")

    r2, z, z_abs = evaluate_model_by_times(df_test, saver)
    print(f"Test: R2: {r2:.5f}, z: {z:.5f}, z_abs: {z_abs:.5f}")
    saver.log(f"Test: R2: {r2:.5f}, z: {z:.5f}, z_abs: {z_abs:.5f}")




def evaluate_model_by_times(df, saver=None):
    df['is_first_client_order'] = 1
    df['is_paid'] = df.is_paid_order

    times = df.groupby('dt_order_placed').dt_order_placed.max().to_dict().values()

    plt_2_gif = Plt2GIF()
    z_score = 0
    z_score_abs = 0
    r2_score = 0

    for t in times:
        df_test = df[df.dt_order_placed == t]
        df_eval = evaluate_by_chunks(df_test)

        plt.cla()
        axes = plt.gca()
        axes.set_xlim([0, df.observed_p2p.quantile(0.95)])
        axes.set_ylim([0, df.p2p_proba.quantile(0.95)])
        plot_qq_plot(df_eval)
        plt_2_gif.add_image(plt)
        plt.cla()

        # plot_z_score_distrib(df_eval)
        # plot_z_score_abs_distrib(df_eval)

        # print(f"time: {t}, chunks R^2 score (corr): {get_r2_coeff(df_eval):.5f}")
        # print(f"time: {t}, mean chunks z-score: {df_eval.z_score.mean():.5f}")
        # print(f"time: {t}, mean chunks abs(z-score): {df_eval.z_score_abs.mean():.5f}\n")

        # saver.log(f"time: {t}, chunks R^2 score (corr): {get_r2_coeff(df_eval):.5f}")
        # saver.log(f"time: {t}, mean chunks z-score: {df_eval.z_score.mean():.5f}")
        # saver.log(f"time: {t}, mean chunks abs(z-score): {df_eval.z_score_abs.mean():.5f}\n")

        r2 = get_r2_coeff(df_eval)
        r2_score += 0 if np.isnan(r2) else r2
        z_score += df_eval.z_score.mean()
        z_score_abs += df_eval.z_score_abs.mean()

    plt_2_gif.make_gif('{}/3_qq_plot.gif'.format(saver.get_folder()))

    return r2_score / len(times), z_score / len(times), z_score_abs / len(times),


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

    mins = df.groupby('dt_order_placed').dt_order_placed.apply(lambda x: x.iloc[0])
    for min in mins:
        axes = plt.gca()
        axes.set_xlim([0, 1])
        check_df = df[df.dt_order_placed == min]
        unpaid_ids = check_df.index.unique()

        paid_ids = df[(df.dt_order_placed < min) & (df.is_paid_order == 1) & (~df.index.isin(unpaid_ids))].index.unique()
        paid_orders = df[df.index.isin(paid_ids)].groupby('order_id').p2p_proba.mean().to_frame()
        paid_orders.p2p_proba = 1
        check_df = check_df.append(paid_orders)
        check_df.p2p_proba.hist(bins=30)
        plt.title('Time: {}, paid orders: {}, unpaid: {}'.format(min, len(paid_orders), len(unpaid_ids)))
        plt_2_gif.add_image(plt)
        # plt.show()
        plt.clf()

    plt_2_gif.make_gif(gif_name, duration=duration)

from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.expression import Insert
@compiles(Insert)
def append_string(insert, compiler, **kw):
    s = compiler.visit_insert(insert, **kw)
    if 'append_string' in insert.kwargs:
        return s + " " + insert.kwargs['append_string']
    return s

def save_features(features_df:pd.DataFrame):
    conn = sqlalchemy.create_engine('mysql+pymysql://root@localhost/edusson_data_science')

    # files = ['features_df_with_web_25000_orders_0_.pkl']
    # files += ['features_df_with_web_25000_orders_25000.pkl']
    files = ['features_df_with_web_25000_orders_50000.pkl']
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

            # conn.execute("SET time_zone='+00:00';")

            try:
                result = conn.execute(ins, values)
            except Exception as e:
                print('Exception', e)
                raise e

            print('Done', end=' ')
            print(i, 'len', len(t_df), end=' ')
            print(str(result), end=' ')
            print('Time:', time()-stime)





if __name__ == '__main__':
    features_df = pd.read_pickle('/home/andrei/Python/sqlalchemy-lab/features_df_with_web_488694.pkl')
    df_train, df_test = get_train_test_features(features_df)

    model = train_model(df_train)
    evaluate_model(model, df_train, df_test)

    FEATURES = ['messages_count', 'edits_count', 'writer_approved_count', 'canceled_bids_count', 'paid_order_count', 'chat_count', 'dt_order_placed_secs', 'dt_last_bid_cancel_secs']
    model = train_model(df_train)
    evaluate_model(model, df_train, df_test)

    FEATURES = ['messages_count', 'edits_count', 'writer_approved_count', 'canceled_bids_count', 'paid_order_count', 'chat_count', 'dt_order_placed_secs', 'dt_last_edit_secs']
    model = train_model(df_train)
    evaluate_model(model, df_train, df_test)

    FEATURES = ['messages_count', 'edits_count', 'writer_approved_count', 'canceled_bids_count', 'paid_order_count', 'chat_count', 'dt_order_placed_secs', 'dt_last_edit_secs', 'dt_last_bid_cancel_secs']
    model = train_model(df_train)
    evaluate_model(model, df_train, df_test)

# DT_FEATURES = ['dt_last_bid_cancel', 'dt_last_chat', 'dt_last_edit', 'dt_last_message', 'dt_last_writer_approved', 'dt_last_paid_order']
# COUNT_FEATURES = ['canceled_bids_count', 'chat_count', 'edits_count', 'messages_count', 'paid_order_count', 'writer_approved_count']