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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

os.environ['DB_ENV'] = 'prod'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.extend(['/home/andrei/Python/sqlalchemy-lab/p2paid'])
from utils import get_actions, fill_empty_values, convert_datetimes_to_seconds, get_dataset, get_adaptive_dataset, get_adaptive_X
import concurent_utils

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


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


print('Start calc times_df')
times_count = 51
actions_df = get_actions(df0)
actions_df = actions_df.set_index('order_id')
actions_df = actions_df.join(df.drop_duplicates('order_id').set_index('order_id').is_paid_order)
times_df = actions_df[actions_df.dt_order_placed != timedelta(seconds=0)].dt_order_placed.quantile(q=np.linspace(0, 1, times_count))
times_df = times_df.reset_index()[:times_count-1].dt_order_placed
times_df.iloc[0] = timedelta()
times_df.to_pickle('times_df_for_all_actions.pkl')

concurent_utils.times_df = times_df
print('Done times_df')

size = 25000
for i in range(0, len(df0), size):
    df = df0[i:i+size]
    print('DF len', len(df))
    print('Get actions')
    stime = time()
    actions_df = get_actions(df)
    actions_df = actions_df.set_index('order_id')
    actions_df = actions_df.join(df.drop_duplicates('order_id').set_index('order_id').is_paid_order)

    print('Done', time()-stime, 'len', len(actions_df))

    # times_count = 51
    #
    # times_df = actions_df[actions_df.dt_order_placed != timedelta(seconds=0)].dt_order_placed.quantile(q=np.linspace(0, 1, times_count))
    # times_df = times_df.reset_index()[:times_count-1].dt_order_placed
    # times_df.iloc[0] = timedelta()

    print('start get_order_features for 4 processes', len(actions_df))
    partial_ids = np.array_split(actions_df.index.unique(), 4)
    partial_dfs = [actions_df[actions_df.index.isin(slice_ids)] for slice_ids in partial_ids]

    with futures.ProcessPoolExecutor() as pool:
        partial_features = pool.map(concurent_utils.process_partial_df, partial_dfs)

    features_df = pd.concat(list(partial_features))
    print('Done', time()-stime)


    print('Save', len(features_df))
    features_df.to_pickle('features_df_with_web_{}_orders_{}.pkl'.format(len(df), i))
