import multiprocessing
import os
from concurrent import futures
from time import time

import sqlalchemy
from sklearn.pipeline import Pipeline
import pandas as pd
from datetime import timedelta
import numpy as np
# from edusson_ds_main.db.connections import DBConnectionsFacade
#
# db_engine_edusson_replica = DBConnectionsFacade.get_edusson_replica()
db_engine_edusson_replica = sqlalchemy.create_engine('mysql+pymysql://code_select:9_4vdmIedhP@edusson-db-replica-2xlarge.cgyy1w9v9yq6.us-east-1.rds.amazonaws.com/edusson')

# id_to_actions_count = {0: 'order_placed', 3: 'writer_approved_count', 5: 'paid_order_count', }
# id_to_last_actions_dt = {0: 'dt_order_placed', 3: 'dt_last_writer_approved', 5: 'dt_last_paid_order'}

id_to_actions_count = {0: 'order_placed', 1: 'messages_count', 2: 'edits_count', 3: 'writer_approved_count',
                       4: 'canceled_bids_count', 5: 'paid_order_count', 6: 'chat_count'}
id_to_last_actions_dt = {0: 'dt_order_placed', 1: 'dt_last_message', 2: 'dt_last_edit', 3: 'dt_last_writer_approved',
                         4: 'dt_last_bid_cancel', 5: 'dt_last_paid_order', 6: 'dt_last_chat'}


def assign_order(x):
    x.iloc[0] = 0
    return x


def get_actions(df):
    ids = df.order_id.unique()

    sql_query = """
    SELECT DISTINCT chat.order_id, chat_message.chat_id, chat_message.date_send FROM es_chat chat
    LEFT JOIN es_chat_message chat_message ON chat_message.chat_id = chat.chat_id AND chat_message.user_sender_id = chat.customer_id
    LEFT JOIN es_orders orders ON orders.order_id = chat.order_id
    WHERE chat.order_id IN ({})
    AND (chat_message.date_send < orders.date_started OR orders.is_paid_order = 0)
    
    AND chat_message.chat_id is not null
    """.format(','.join(ids.astype(str)))

    df_messages = pd.read_sql(sql=sql_query, con=db_engine_edusson_replica)
    df_messages = df_messages.rename({'date_send': 'action_date'}, axis='columns')
    df_messages['action_id'] = 1

    chats_df = df_messages.copy()

    if 'chat_id' in df_messages.columns:
        df_messages = df_messages.drop(columns=['chat_id'])

    sql_query = """
    SELECT DISTINCT order_id, order_date FROM es_orders_audit audit 
    WHERE order_id IN ({})
    """.format(','.join(ids.astype(str)))
    df_edits = pd.read_sql(sql=sql_query, con=db_engine_edusson_replica)
    df_edits = df_edits.rename({'order_date': 'action_date'}, axis='columns')
    df_edits['action_id'] = 2
    df_edits.action_id = df_edits.groupby('order_id').action_id.transform(assign_order)

    ###################### date writer appproved #######################################
    df_writer_approved = df[df.date_writer_approved.notna()][['order_id', 'date_writer_approved']]
    df_writer_approved = df_writer_approved.rename({'date_writer_approved': 'action_date'}, axis='columns')
    df_writer_approved['action_id'] = 3
    ###################### date writer appproved #######################################

    ####################### bids canceled ##############################################
    sql_query = '''
    SELECT tmp.order_id, tmp.date_state_change
             FROM es_bids tmp
             WHERE tmp.order_id in ({}) 
             AND tmp.state_id=4 
             AND tmp.bid_type_id=1
             '''.format(','.join(ids.astype(str)))
    df_canceled_bids = pd.read_sql(sql=sql_query, con=db_engine_edusson_replica)
    df_canceled_bids = df_canceled_bids.rename({'date_state_change': 'action_date'}, axis='columns')
    df_canceled_bids['action_id'] = 4
    ####################### bids canceled ##############################################

    ###################### date paid #######################################
    df_paid = df[df.date_paid.notna()][['order_id', 'date_paid']]
    df_paid = df_paid.rename({'date_paid': 'action_date'}, axis='columns')
    df_paid['action_id'] = 5
    ###################### date paid #######################################

    ###################### date chat #######################################

    if not chats_df.empty:
        chats_df = chats_df.groupby(['order_id', 'chat_id']).action_date.min().to_frame()

        # chat_id can be None (fixed by sql where clause)
        # if not chats_df.empty:
        chats_df.reset_index(inplace=True)
        chats_df = chats_df.drop(columns=['chat_id'])
        chats_df['action_id'] = 6
    ###################### date chat #######################################

    df_actions = pd.concat((df_messages, df_edits, df_writer_approved, df_canceled_bids, df_paid, chats_df))
    df_actions = df_actions.reset_index(drop=True)
    df_actions = df_actions.sort_values('action_date')
    df_actions['dt_order_placed'] = df_actions.groupby('order_id').action_date.apply(lambda x: x - x.min())

    df_actions = df_actions[df_actions.action_id.isin(list(id_to_actions_count.keys()))].copy()

    return df_actions


def get_features_from_actions(df):
    return df.groupby('order_id').apply(get_order_features)

# def get_features_from_actions(df):
#     return apply_by_multiprocessing(df.groupby('order_id'), get_order_features, workers=4)



def get_order_features(df):
    return pd.concat([get_features(df, dt_order_placed) for dt_order_placed in df.dt_order_placed])

# def get_order_features(df):
#     with futures.ProcessPoolExecutor() as pool:
#         times = [dt_order_placed for dt_order_placed in df.dt_order_placed]
#         res = pool.map(get_features, [df] * len(times), times, chunksize=100)
#     return pd.concat(res)


# name_features = apply_by_multiprocessing(df.tokenized_name,
#                                          make_categorical_features,
#                                          workers=4)


def _apply_df(args):
    df, func, kwargs = args
    return df.apply(func, **kwargs)


def apply_by_multiprocessing(df, func, **kwargs):
    workers = kwargs.pop('workers')
    pool = multiprocessing.Pool(processes=workers)
    result = pool.map(_apply_df, [(d, func, kwargs)
                                  for d in np.array_split(df, workers)])
    pool.close()
    return pd.concat(list(result))






def get_features(df, dt_order_placed=timedelta(minutes=1)):
    df = df[~df.action_id.isin([0, 5])].copy()
    df_actions_count = df.groupby('action_id').dt_order_placed.apply(lambda x: (x <= dt_order_placed).sum())
    df_actions_count = df_actions_count.rename(id_to_actions_count)

    df_last_action_dt = df.groupby('action_id').dt_order_placed.apply(lambda x: x[x <= dt_order_placed].max())
    mask = df_last_action_dt.notna()

    if mask.sum():
        df_last_action_dt.loc[mask] = dt_order_placed - df_last_action_dt[mask]

    df_last_action_dt = df_last_action_dt.rename(id_to_last_actions_dt)

    df_features = pd.concat((df_actions_count, df_last_action_dt))
    df_features.index.name = ''
    df_features = df_features.to_frame(dt_order_placed).T
    df_features.index.name = 'dt_order_placed'
    return df_features


def get_structured_df_from_actions(actions_df, slice_dt=timedelta(days=2), original_df=None):
    features_df = get_features_from_actions(actions_df)

    # features_df = apply_by_multiprocessing(actions_df, get_features_from_actions, workers=4)

    features_df.reset_index(inplace=True)
    features_df = features_df.set_index('order_id')
    # features_df['is_paid_order'] = original_df.is_paid_order
    features_df = features_df.join(actions_df.is_paid_order)
    features_df = features_df.drop_duplicates()

    mask_paid_more_2_days = (actions_df.is_paid_order == 1) & (actions_df.dt_order_placed > slice_dt)
    mask_non_paid = (actions_df.is_paid_order == 0)

    # remove all action that more than slice dt limit
    features_df = features_df[features_df.dt_order_placed <= slice_dt]

    features_df.reset_index(inplace=True)
    last_features = actions_df[mask_paid_more_2_days | mask_non_paid].groupby('order_id').apply(lambda x: get_features(x, slice_dt))
    last_features.reset_index(inplace=True)

    features_df = features_df.append(last_features)

    # fill last_features rows with is_paid_order
    features_df.is_paid_order = features_df.is_paid_order.fillna(0)
    features_df = features_df.set_index('order_id')
    features_df.is_paid_order = features_df.groupby('order_id').is_paid_order.apply(lambda x: x.max())

    # features_df = features_df.set_index('order_id')
    features_df = features_df.sort_values(['dt_order_placed'])
    return features_df


def convert_datetimes_to_seconds(df, d_dt=None):
    keys = list(id_to_last_actions_dt.values())
    keys = [key for key in keys if key in df.columns]
    df[keys] = df[keys].apply(lambda x: x.dt.total_seconds())
    return df



def fill_empty_values(df, max_fill_dt_value=timedelta(days=30), d_cnt=None, d_dt=None):
    keys = list(id_to_actions_count.values())
    keys = [key for key in keys if key in df.columns]
    df[keys] = df[keys].fillna(0)

    keys = list(id_to_last_actions_dt.values())
    keys = [key for key in keys if key in df.columns]
    df[keys] = df[keys].fillna(max_fill_dt_value)

    return df


def get_structured_df(df):
    print('Get actions', 'len df', len(df), 'pid:', os.getpid())
    stime = time()
    actions_df = get_actions(df)
    actions_df = actions_df.set_index('order_id')
    actions_df['is_paid_order'] = df.set_index('order_id').is_paid_order
    print('Done', time()-stime, 'pid:', os.getpid())

    print('Get structured_df_from_actions', len(actions_df), 'pid:', os.getpid())
    stime = time()
    structured_df = get_structured_df_from_actions(actions_df)
    structured_df.is_paid_order = structured_df.is_paid_order.fillna(0)
    print('Done', time()-stime, 'pid:', os.getpid())

    return structured_df


def get_dataset(df, keys=None):
    if not keys:
        count_keys = list(id_to_actions_count.values())
        count_keys.remove('paid_order_count')
        count_keys.remove('order_placed')
        dt_keys = list(id_to_last_actions_dt.values())
        dt_keys.remove('dt_last_paid_order')
        # dt_keys.remove('dt_last_writer_approved')
        keys = count_keys + dt_keys
    label = 'is_paid_order'

    not_exists_keys = [key for key in keys if key not in df.columns]
    keys = [key for key in keys if key in df.columns]

    # df[not_exists_keys] = 0

    X = df[keys].values
    Y = df[label].values
    return X, Y


def get_adaptive_dataset(df, keys=None):
    label = 'is_paid_order'

    not_exists_keys = [key for key in keys if key not in df.columns]


    for key in not_exists_keys:
        df[key] = 0

    keys = [key for key in keys if key in df.columns]

    X = df[keys].values
    Y = df[label].values
    return X, Y



def get_features_by_time(actions_df, dt=timedelta(minutes=1)):
    df = actions_df.groupby('order_id').apply(lambda x: get_features(x, dt))
    df.reset_index(inplace=True)
    df = df.set_index('order_id')
    return prepare_dataset(df)


def prepare_dataset(df):
    df = fill_empty_values(df)
    return convert_datetimes_to_seconds(df)



class OnlinePipeline(Pipeline):
    def partial_fit(self, X, y=None):
        for i, step in enumerate(self.steps):
            name, est = step
            est.partial_fit(X, y)
            if i < len(self.steps) - 1:
                X = est.transform(X)
        return self
