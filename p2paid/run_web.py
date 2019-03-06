#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 15:33:19 2019
@author: luchkoivan
[order_id, timesamp, action]
FEATURES:
    order_placed_date ==>> dt_order_placed

    client_chats_count
    date_last_client_chat ==>> dt_last_client_chat

    client_msgs_count
    date_last_client_msg ==>> dt_last_client_msg

    order_edit_count
    date_last_edit ==>> dt_last_edit

    writer_approved
    date_writer_approved ==>> dt_writer_approved

    ------
    place2paid_proba_placed  # initial, after order placed
    active_bids_count   # dynamic / same as other actions
    date_last_bid_placed ==>> dt_last_bid_placed
"""
import cProfile
from concurrent import futures
from datetime import timedelta

import os
from time import time

import sqlalchemy
from sklearn.pipeline import Pipeline

os.environ['DB_ENV'] = 'prod'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.extend(['D:\Python\sqlalchemy-lab\p2paid'])
from utils import get_actions, fill_empty_values, convert_datetimes_to_seconds, get_dataset, get_adaptive_dataset

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
pr = cProfile.Profile()


#from edusson_ds_main.db.connections import DBConnectionsFacade

#db_engine_edusson_replica = DBConnectionsFacade.get_edusson_replica()
db_engine_edusson_replica = sqlalchemy.create_engine('mysql+pymysql://code_select:9_4vdmIedhP@edusson-db-replica-2xlarge.cgyy1w9v9yq6.us-east-1.rds.amazonaws.com/edusson')

id_to_actions_count = {0: 'order_placed', 1: 'messages_count', 2: 'edits_count', 3: 'writer_approved_count', 4: 'canceled_bids_count', 5: 'paid_order_count', 6: 'chat_count'}
id_to_last_actions_dt = {0: 'dt_order_placed', 1: 'dt_last_message', 2: 'dt_last_edit', 3: 'dt_last_writer_approved', 4: 'dt_last_bid_cancel', 5: 'dt_last_paid_order', 6: 'dt_last_chat'}

pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 2000)
pd.set_option('display.line_width', 2000)

sql_query = """
    SELECT  
        t0.order_id,
        t0.is_paid_order,
        t0.date_started AS date_paid,
        bids.date_state_change AS date_writer_approved,
        order_additional.smart_bidding_place2paid AS place2paid_proba
        
    FROM es_orders t0
    LEFT JOIN es_orders_additional order_additional ON order_additional.order_id = t0.order_id
    LEFT JOIN es_bids bids ON bids.order_id = t0.order_id AND bids.state_id = 5 AND bid_type_id = 1
    
    WHERE t0.order_date > '2018-01-01' 
    AND t0.is_easy_bidding = 0
    AND t0.is_first_client_order = 1
    AND order_additional.device_type_id_create = 1
    AND t0.order_id NOT IN (SELECT order_id FROM es_order_reassign_history)
    AND t0.test_order = 0
    AND t0.site_id != 31 
    ORDER BY RAND()"""

df0 = pd.read_sql(sql=sql_query, con=db_engine_edusson_replica)
df = df0.copy()

df = df0[:30000]
print('DF len', len(df))


# df = df[df.order_id.isin([1088491, 1058728, 1505552, 1494183])]

################################ train ##########################################################
print('Get actions')
stime = time()
actions_df = get_actions(df)
actions_df = actions_df.set_index('order_id')
actions_df['is_paid_order'] = df.set_index('order_id').is_paid_order
print('Done', time()-stime, 'len', len(actions_df))



times = [timedelta(minutes=i) for i in range(120)]
# slice_dt = timedelta(minutes=25)


def get_order_features(df):
    return pd.concat([get_features(df, dt_order_placed) for dt_order_placed in times])
    # return pd.concat([get_features(df, dt_order_placed) for dt_order_placed in times if dt.empty or dt.item() <= dt_order_placed])
    # return pd.concat([get_features(df, dt_order_placed) for dt_order_placed in df.dt_order_placed])


def get_features(df, dt_order_placed=timedelta(minutes=1)):
    df = df[~df.action_id.isin([0])].copy()
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


features_df = actions_df.groupby('order_id').apply(get_order_features)
print('Save', len(features_df))
features_df.to_pickle('features_df_with_web.pkl')


df[df.order_id == 1088491]
actions_df.loc[1088491]
features_df.loc[1088491]



features_df.loc[1058728]
features_df.loc[1088491]
features_df.loc[1494183]
features_df.loc[1505552]

features_df.index.unique()

features_df = features_df[features_df.dt_last_paid_order.isna()]







features_df = features_df.reset_index().set_index('order_id')


features_df['is_paid_order'] = df.set_index('order_id').is_paid_order
features_df = features_df.drop(columns=['dt_last_paid_order', 'paid_order_count'])


original_df_mean = df.is_paid_order.mean()
features_df_mean = features_df.groupby('order_id').is_paid_order.apply(lambda x: x.max()).mean()
assert(original_df_mean == features_df_mean), "original_df_mean AND features_df_mean does not equal"


mask_paid_more_2_days = (actions_df.is_paid_order == True) & (actions_df.dt_order_placed > slice_dt)
mask_non_paid = (actions_df.is_paid_order == False)

# remove all action that more than slice dt limit
features_df = features_df[features_df.dt_order_placed <= slice_dt]

features_df = features_df.reset_index().set_index('order_id')
last_features = actions_df[mask_paid_more_2_days | mask_non_paid].groupby('order_id').apply(lambda x: get_features(x, slice_dt))
last_features = last_features.reset_index().set_index('order_id')


feat = features_df.append(last_features)
feat.is_paid_order = feat.is_paid_order.fillna(False)
feat.is_paid_order = feat.groupby('order_id').is_paid_order.apply(lambda x: x.max())


# feat = feat.drop(columns=['dt_last_paid_order', 'paid_order_count'])
feat.head()

original_df_mean = df.is_paid_order.mean()
features_df_mean = feat.groupby('order_id').is_paid_order.apply(lambda x: x.max()).mean()
# assert(original_df_mean == features_df_mean), "original_df_mean AND features_df_mean does not equal"

# stored_df = feat.copy()
feat = feat.drop(columns=['dt_last_chat', 'dt_last_edit', 'dt_last_message', 'dt_last_paid_order', 'dt_last_writer_approved'])
# feat = feat.drop(columns=['dt_last_bid_cancel', 'dt_last_chat', 'dt_last_edit', 'dt_last_message', 'dt_last_paid_order', 'dt_last_writer_approved'])

d_cnt = ['messages_count', 'edits_count', 'writer_approved_count', 'canceled_bids_count', 'paid_order_count', 'chat_count']
d_dt = ['dt_order_placed']

feat_filled = fill_empty_values(feat.copy(), d_cnt=d_cnt, d_dt=d_dt)
feat_seconds = convert_datetimes_to_seconds(feat_filled.copy(), d_dt=d_dt)
feat_seconds = feat_seconds.sample(frac=1)
X, Y = get_adaptive_dataset(feat_seconds, d_cnt+d_dt)

X.shape


scaler = StandardScaler()

mlp = MLPClassifier(hidden_layer_sizes=(3, 3),
                    max_iter=1000, verbose=10, tol=1e-6,
                    activation='tanh', solver='adam')

piple = Pipeline([
    ('scale', StandardScaler()),
    ('mlp', mlp),
])

piple.fit(X, Y) # can be passed generator



y_pred_proba = piple.predict_proba(X)
print('Train Predicted', y_pred_proba.mean(0), 'Actual', features_df_mean)




test_actions_df = get_actions(df0[:30000])
test_actions_df = test_actions_df.set_index('order_id')
# test_actions_df['is_paid_order'] = df0.set_index('order_id').is_paid_order
test_actions_df = test_actions_df.join(df0.set_index('order_id').is_paid_order)
test_actions_df = test_actions_df.drop_duplicates()

id_to_actions_count = {0: 'order_placed', 1: 'messages_count', 2: 'edits_count', 3: 'writer_approved_count',
                       4: 'canceled_bids_count', 5: 'paid_order_count', 6: 'chat_count'}
id_to_last_actions_dt = {0: 'dt_order_placed', 1: 'dt_last_message', 2: 'dt_last_edit', 3: 'dt_last_writer_approved',
                         4: 'dt_last_bid_cancel', 5: 'dt_last_paid_order', 6: 'dt_last_chat'}
exclude_ids = []


test_predict_actions_df = test_actions_df[~test_actions_df.action_id.isin(exclude_ids)]

test_predict_actions_df = test_predict_actions_df.sample(frac=1)

df0[:30000].is_paid_order.mean()
test_predict_actions_df[:30000].copy().groupby('order_id').is_paid_order.apply(lambda x: x.max()).mean()

def graph(actions_df, times):
    x_g = []

    for i in times:
        dt = timedelta(minutes=i)

        predicted = predict_by_time(actions_df, dt)

        # test_features_df_mean = t_df.groupby('order_id').is_paid_order.apply(lambda x: x.max()).mean()
        # print('Test Predicted for', dt, y_test_pred_proba.mean(0), 'Actual', test_features_df_mean)
        x_g.append(predicted)

    return x_g

def predict_by_time(df, dt):
    t_df = df.groupby('order_id').apply(lambda x: get_features(x, dt))
    t_df = t_df.reset_index().set_index('order_id')
    t_df = t_df.join(df0.set_index('order_id').is_paid_order)
    t_df = t_df.drop_duplicates()

    test_feat_filled = fill_empty_values(t_df)

    test_feat_seconds = convert_datetimes_to_seconds(test_feat_filled, d_dt=d_dt)
    test_X, test_Y = get_adaptive_dataset(test_feat_seconds, d_cnt+d_dt)

    y_test_pred_proba = piple.predict_proba(test_X)
    # test_features_df_mean = t_df.groupby('order_id').is_paid_order.apply(lambda x: x.max()).mean()
    return y_test_pred_proba.mean(0)[1]

predict_by_time(test_predict_actions_df[:1000].copy(), timedelta(minutes=160))



y_g = list(range(60))
x_g = graph(test_predict_actions_df[:500].copy(), y_g)

plt.plot(y_g, x_g)
plt.show()
