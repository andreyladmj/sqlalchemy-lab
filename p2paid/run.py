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
from datetime import timedelta

import os
from time import time

import sqlalchemy

os.environ['DB_ENV'] = 'prod'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from p2paid.utils import get_actions, get_structured_df_from_actions, prepare_dataset, get_dataset, OnlinePipeline, \
    get_features_by_time

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelBinarizer
import matplotlib.pyplot as plt
import seaborn as sns
pr = cProfile.Profile()

db_engine_edusson_replica = sqlalchemy.create_engine('mysql+pymysql://code_select:9_4vdmIedhP@edusson-db-replica-2xlarge.cgyy1w9v9yq6.us-east-1.rds.amazonaws.com/edusson')

id_to_actions_count = {0: 'order_placed', 1: 'messages_count', 2: 'edits_count', 3: 'writer_approved_count', 4: 'canceled_bids_count', 5: 'paid_order_count', 6: 'chat_count'}
id_to_last_actions_dt = {0: 'dt_order_placed', 1: 'dt_last_message', 2: 'dt_last_edit', 3: 'dt_last_writer_approved', 4: 'dt_last_bid_cancel', 5: 'dt_last_paid_order', 6: 'dt_last_chat'}

pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
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
    AND t0.site_id != 31 ORDER BY RAND()"""

df0 = pd.read_sql(sql=sql_query, con=db_engine_edusson_replica)
df = df0.copy()

df = df0[:50000]
test_df = df0[50000:60000]
bastch_size = 1024
print('DF len', len(df))
print('Test DF len', len(test_df))


scaler = StandardScaler()

mlp = MLPClassifier(hidden_layer_sizes=(15, 10, 10, 5),
                    max_iter=1000, verbose=10, tol=1e-5,
                    activation='tanh', solver='adam')

piple = OnlinePipeline([
    ('scale', StandardScaler()),
    ('mlp', mlp),
])


# train(df)
################################ train ##########################################################
print('Get actions')
stime = time()
actions_df = get_actions(df)
actions_df = actions_df.set_index('order_id')
actions_df['is_paid_order'] = df.set_index('order_id').is_paid_order
print('Done', time()-stime)

print('Get structured_df_from_actions', len(actions_df))
stime = time()
structured_df = get_structured_df_from_actions(actions_df)
structured_df.is_paid_order = structured_df.is_paid_order.fillna(0)
print('Done', time()-stime)

print('structured_df.is_paid_order.mean()', structured_df.is_paid_order.mean())

print('Prepare_dataset')
stime = time()
featured_df_with_seconds = prepare_dataset(structured_df)
print('Done', time()-stime)

# shuffle
featured_df_with_seconds = featured_df_with_seconds.sample(frac=1)

X, Y = get_dataset(featured_df_with_seconds)
print('partial_fit', 'X shape', X.shape, 'Y.shape', Y.shape, 'actions_df', len(actions_df))

piple.fit(X, Y)
# test_prediction(test_df, times=[0, 60, 120, hour_1, days_2])
####################################################################
# y_pred_proba0 = piple.predict_proba(test_X0)
# print('Predict Proba for time {}s mean'.format(0), y_pred_proba0.mean(0), 'df place2paid_proba mean', test_df.place2paid_proba.mean(), 'is_paid order', test_df.is_paid_order.mean())
####################################################################

# structured_df.is_paid_order.mean()
# df.is_paid_order.mean()


################################ test ##########################################################

hour_1 = timedelta(hours=1).total_seconds()
days_2 = timedelta(days=2).total_seconds()
times=[0, 60, 120, hour_1, days_2]
test_actions_df = get_actions(test_df)
test_actions_df = test_actions_df.set_index('order_id')
test_actions_df = test_actions_df.join(test_df.set_index('order_id').is_paid_order)

print('df.is_paid_order.mean()', df.is_paid_order.mean())

##################### test for 0 secs ##########################################
features0 = get_features_by_time(test_actions_df, timedelta(seconds=0))
features0 = features0.join(test_actions_df.is_paid_order)
test_X0, test_Y0 = get_dataset(features0)

# y_pred = piple.predict(test_X)
y_pred_proba0 = piple.predict_proba(test_X0)
print('Predict Proba for time {}s mean'.format(0), y_pred_proba0.mean(0), 'df place2paid_proba mean', test_df.place2paid_proba.mean(), 'is_paid order', test_df.is_paid_order.mean())
################################################################################





print('len test_actions_df', len(test_actions_df))
for secs in times:
    features = get_features_by_time(test_actions_df, timedelta(seconds=secs))
    features = features.join(test_actions_df.is_paid_order)
    test_X, test_Y = get_dataset(features)

    # y_pred = piple.predict(test_X)
    y_pred_proba = piple.predict_proba(test_X)
    print('Predict Proba for time {}s mean'.format(secs), y_pred_proba.mean(0), 'df place2paid_proba mean', test_df.place2paid_proba.mean(), 'is_paid order', test_df.is_paid_order.mean())


################################ graph ################################################################
graph_df_size = 100

# print('len test_actions_df', len(test_actions_df))
# # x_plt = []
# y_plt = list(range(60*3))
#
# def predict_for_times(mins):
#     features = get_features_by_time(test_actions_df[:graph_df_size], timedelta(minutes=mins))
#     test_X, test_Y = get_dataset(features, original_df=df)
#     y_pred_proba = piple.predict_proba(test_X)
#     return y_pred_proba.mean(0)[1]
#
# from concurrent import futures
#
# with futures.ProcessPoolExecutor() as pool:
#     x_plt = pool.map(predict_for_times, y_plt)
#     plt.plot(y_plt, x_plt)
#     plt.show()
y_plt = list(range(60*19))
x_plt = []

for mins in y_plt:
    features = get_features_by_time(test_actions_df[:graph_df_size], timedelta(minutes=mins))
    test_X, test_Y = get_dataset(features, original_df=df)

    y_pred = piple.predict(test_X)
    y_pred_proba = piple.predict_proba(test_X)
    predicts = y_pred_proba.mean(0)[1]
    x_plt.append(predicts)

plt.plot(y_plt, x_plt)
plt.show()