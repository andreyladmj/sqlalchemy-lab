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
from utils import get_bidding_events, fill_empty_values, convert_datetimes_to_seconds, get_dataset, get_adaptive_dataset, get_adaptive_X, get_orders_info
import concurent_utils


#db_engine_edusson_replica = DBConnectionsFacade.get_edusson_replica()

pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 2000)
pd.set_option('display.line_width', 2000)

df0 = get_orders_info()
df = df0.copy()

# jjj = df.sort_values('date_paid')[:100]
# get_bidding_events(df[df.order_id == 627476])
# get_bidding_events(jjj).event_id.value_counts()

# er = df_events.resample('W', on='event_date').apply(lambda x: (x.event_id == 1).sum() / x.order_id.nunique())
# er.plot()
# plt.show()

# conn = sqlalchemy.create_engine('mysql+pymysql://root@localhost/edusson_data_science')
#
# stime = time()
# df = pd.read_sql(sql='SELECT * from p2p_order_features WHERE dt_order_placed <=11782 limit 1000000;', con=conn)
# print('End', time()-stime)

# df = df[df.dt_order_placed <= 11782]
# for col in df.columns:
#     df[col].hist(bins=40)
#     plt.title(col)
#     plt.show()




print('Start calc times_df')
times_count = 101
df_events = pd.concat([get_bidding_events(part_df) for part_df in np.array_split(df, 4)])
df_events.to_pickle('all_events.pkl')
times_df = df_events[df_events.dt_order_placed != timedelta(seconds=0)].dt_order_placed.quantile(q=np.linspace(0, 1, times_count))
times_df = times_df.reset_index()[:times_count-1].dt_order_placed
times_df.iloc[0] = timedelta()
times_df = times_df.apply(lambda delta: timedelta(seconds=round(delta.total_seconds())))
times_df.to_pickle('times_df_for_all_events.pkl')
# times_df = pd.read_pickle('times_df_for_all_events.pkl')
concurent_utils.times_df = times_df
print('Done times_df')

size = 25000
for iteration, i in enumerate(range(0, len(df0), size)):
    df_orders = df0[i:i+size]
    print('DF len', len(df_orders))
    print('Get events')
    stime = time()
    df_events = get_bidding_events(df_orders)
    # df_events = df_events.set_index('order_id')
    # df_events = df_events.join(df_orders.drop_duplicates('order_id').set_index('order_id').is_paid_order)
    print('Done', time()-stime, 'len', len(df_events))

    print('start get_order_features for 4 processes', len(df_events))
    partial_ids = np.array_split(df_events.order_id.unique(), 4)
    partial_dfs = [df_events[df_events.order_id.isin(slice_ids)] for slice_ids in partial_ids]

    with futures.ProcessPoolExecutor() as pool:
        partial_features = pool.map(concurent_utils.process_partial_df, partial_dfs)

    features_df = pd.concat(list(partial_features))
    print('Done', time()-stime)

    # features_df.reset_index().order_id.nunique()
    # test_df = features_df.reset_index()
    # test_df[test_df.order_id == 1427287]

    print('Save', len(features_df))
    features_df.to_pickle('features_{}_orders_{}.pkl'.format(iteration, len(df_orders)))
