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
# db_engine_edusson_ds_replica = sqlalchemy.create_engine('mysql+pymysql://developer:9_4vdmIedhP@edusson-data-science-db.cgyy1w9v9yq6.us-east-1.rds.amazonaws.com/edusson')

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

df = df0[:50000]
print('DF len', len(df))


# df = df[df.order_id.isin([1088491, 1058728, 1505552, 1494183])]

################################ train ##########################################################
print('Get actions')
stime = time()
actions_df = get_actions(df)
actions_df = actions_df.set_index('order_id')
# actions_df['is_paid_order'] = df.set_index('order_id').is_paid_order
# actions_df = actions_df.join(df.set_index('order_id').is_paid_order).drop_duplicates('order_id')
actions_df = actions_df.join(df.drop_duplicates('order_id').set_index('order_id').is_paid_order)


print('Done', time()-stime, 'len', len(actions_df))

times_count = 21

times_df = actions_df[actions_df.dt_order_placed != timedelta(seconds=0)].dt_order_placed.quantile(q=np.linspace(0, 1, times_count))
times_df = times_df.reset_index()[:times_count-1].dt_order_placed
times_df.iloc[0] = timedelta()
# times_df.to_pickle('times_df.pkl')

concurent_utils.times_df = times_df

# try:
print('start get_order_features for 4 processes', len(actions_df))
partial_ids = np.array_split(actions_df.index.unique(), 4)
partial_dfs = [actions_df[actions_df.index.isin(slice_ids)] for slice_ids in partial_ids]

with futures.ProcessPoolExecutor() as pool:
    partial_features = pool.map(concurent_utils.process_partial_df, partial_dfs)

features_df = pd.concat(list(partial_features))
print('Done', time()-stime)
# start get_order_features for 4 processes 281809
# Done 4260.6323318481445

# except Exception as e:
#     print('WARNING!', str(e))
#
#     slice_ids = actions_df.index.unique()[:25000]
#     actions_df = actions_df[actions_df.index.isin(slice_ids)]
#
#     print('start get_order_features', len(actions_df))
#     stime = time()
#     features_df = actions_df.groupby('order_id').apply(get_order_features)  # 129942 rows for 3569s
#     print('Done', time()-stime)

print('Save', len(features_df))
features_df.to_pickle('features_df_with_web_{}_orders.pkl'.format(len(df)))

# features_df = pd.read_pickle('/home/andrei/Python/sqlalchemy-lab/features_df_with_web_488694.pkl')

features_df = features_df.reset_index().set_index('order_id')
features_df = features_df.join(df.drop_duplicates('order_id').set_index('order_id').is_paid_order)

dt_list = ['dt_last_bid_cancel', 'dt_last_chat', 'dt_last_edit', 'dt_last_message', 'dt_last_writer_approved', 'dt_last_paid_order']

features_df[dt_list] = features_df[dt_list].fillna(timedelta(days=30))


features_df[['{}_secs'.format(key) for key in dt_list]] = features_df[dt_list].apply(lambda x: x.dt.total_seconds())
features_df[['dt_order_placed_secs']] = features_df[['dt_order_placed']].apply(lambda x: x.dt.total_seconds())


# print('describe dt_order_placed')
# print(features_df.dt_order_placed.describe())
# features_df = features_df.drop(columns=['dt_last_paid_order', 'paid_order_count'])


original_df_mean = df.is_paid_order.mean()
features_df_mean = features_df.groupby('order_id').is_paid_order.apply(lambda x: x.max()).mean()
assert(original_df_mean == features_df_mean), "original_df_mean AND features_df_mean does not equal"

FEATURES = ['messages_count', 'edits_count', 'writer_approved_count', 'canceled_bids_count', 'paid_order_count', 'chat_count', 'dt_order_placed_secs']
test_size = 0.2

################### real_p2p_proba ##############################
observed_p2p_df = features_df.groupby('dt_order_placed').is_paid_order.mean().to_frame('observed_p2p')
features_df = features_df.join(observed_p2p_df.observed_p2p, on='dt_order_placed')

features_df = features_df.fillna(0)

df_train, df_test = train_test_split(features_df, test_size=test_size)

scaler = StandardScaler()

mlp = MLPClassifier(hidden_layer_sizes=(32, 16, 16, 8, 4),
                    max_iter=1000, verbose=10, tol=1e-5,
                    activation='tanh', solver='adam')

piple = Pipeline([
    ('scale', StandardScaler()),
    ('mlp', mlp),
])

piple.fit(df_train[FEATURES], df_train.is_paid_order) # can be passed generator



y_pred_proba = piple.predict_proba(df_train[FEATURES])
print('Train Predicted', y_pred_proba.mean(0), 'Actual', features_df_mean)

features_df['p2p_proba'] =5

# df_train.groupby('dt_order_placed').apply(lambda x: piple.predict_proba)
df_train['p2p_proba'] = piple.predict_proba(df_train[FEATURES])[:, 1]



#############################################################
compare_p2p = df_train.groupby('dt_order_placed')[['is_paid_order', 'p2p_proba']].mean()

compare_p2p.index = compare_p2p.index.total_seconds() / 60
compare_p2p.plot()
plt.show()

df_train['is_paid'] = df_train['is_paid_order']
df_train['is_first_client_order'] = 1

df_train.index.nunique()

times_df
# for t in df_train.dt_order_placed.unique():
for t in times_df:
    check_df = df_train[df_train.dt_order_placed == t]
    print(t)
    evaluate(check_df)


'''
10 - 100
5  - 
'''






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




test_actions_df = get_actions(df0)
test_actions_df = test_actions_df.set_index('order_id')
test_actions_df = test_actions_df.join(df0.set_index('order_id').is_paid_order)
test_actions_df = test_actions_df.drop_duplicates()

id_to_actions_count = {0: 'order_placed', 1: 'messages_count', 2: 'edits_count', 3: 'writer_approved_count',
                       4: 'canceled_bids_count', 5: 'paid_order_count', 6: 'chat_count'}
id_to_last_actions_dt = {0: 'dt_order_placed', 1: 'dt_last_message', 2: 'dt_last_edit', 3: 'dt_last_writer_approved',
                         4: 'dt_last_bid_cancel', 5: 'dt_last_paid_order', 6: 'dt_last_chat'}

######################### MAKE HIST #####################################

def pp(df, dt):
    t_df = get_features(df, dt)
    t_df = t_df.reset_index()

    if 'paid_order_count' in t_df.columns and t_df.paid_order_count.max() == 1:
        return 1

    test_feat_filled = fill_empty_values(t_df)
    test_feat_seconds = convert_datetimes_to_seconds(test_feat_filled, d_dt=d_dt)
    test_X = get_adaptive_X(test_feat_seconds, d_cnt+d_dt)

    y_test_pred_proba = piple.predict_proba(test_X)
    # test_features_df_mean = t_df.groupby('order_id').is_paid_order.apply(lambda x: x.max()).mean()
    return y_test_pred_proba.mean(0)[1]


test_actions_df = test_actions_df.sample(frac=1)
u_ids = test_actions_df.index.unique()

hist_df = test_actions_df[test_actions_df.index.isin(u_ids[:10000])]

p_mins = [0, 1, 2, 5, 7, 10, 15, 20, 30, 45, 60, 90]
probs_dfs = []

for mins in p_mins:
    print('mins', mins)

    x_vals = hist_df.groupby('order_id').apply(lambda x: pp(x, timedelta(minutes=mins)))
    # hist_df[15:16].groupby('order_id').apply(lambda x: pp(x, timedelta(minutes=1160)))
    # hist_df.loc[1519039].groupby('order_id').apply(lambda x: pp(x, timedelta(minutes=0)))
    # hist_df.groupby('order_id').apply(lambda x: pp(x, timedelta(minutes=0)))
    # df0[df0.order_id == 1519039]

    fr = x_vals.to_frame('probability')

    fr['is_paid_orders'] = hist_df.groupby('order_id').is_paid_order.max()

    # tmp_df = fr.reset_index().groupby('order_id').apply(lambda x: x[['probability', 'is_paid_orders']].max(axis=1)).to_frame('probability_')
    # tmp_df = tmp_df.reset_index().drop(columns=['level_1']).set_index('order_id')
    # fr['probability_'] = tmp_df.probability_

    probs_dfs.append(fr)

    fr.probability.hist(bins=50)
    plt.title('{} mins'.format(mins))
    plt.show()


probs_dfs[0].probability.mean()
probs_dfs[0].is_paid_orders.mean()

for h, m in zip(probs_dfs, p_mins):

    mdf = h[h.probability != 1]

    print(m, mdf.probability.mean(), mdf.is_paid_orders.mean(), mdf.probability.std())
    # h.probability.hist(bins=20)
    # plt.title('{} mins'.format(m))
    # plt.show()


######################### MAKE HIST #####################################

############################################################################3
#test_actions_df[~test_actions_df.action_id.isin([0,1,5])].action_id.unique()


# actions_1 = [0,1,2,3,4,5,6]
actions_1 = [0,1,5]
remove_ids = test_actions_df[~test_actions_df.action_id.isin(actions_1)].index.unique()
peoples_whos_do_action_1 = test_actions_df[~test_actions_df.index.isin(remove_ids)]

peoples_whos_do_action_1.action_id.unique()



actions_2 = [0,5]
remove_ids = test_actions_df[~test_actions_df.action_id.isin(actions_2)].index.unique()
peoples_whos_do_action_2 = test_actions_df[~test_actions_df.index.isin(remove_ids)]

peoples_whos_do_action_2.action_id.unique()


def graph(actions_df, times):
    x_g = []

    for i in times:
        dt = timedelta(minutes=i)

        predicted = predict_by_time(actions_df, dt)
        print(dt, predicted)
        x_g.append(predicted)

    return x_g


y_g = list(range(0, 60, 2))
x1_g = graph(peoples_whos_do_action_1[:500].copy(), y_g)
x2_g = graph(peoples_whos_do_action_2[:500].copy(), y_g)

plt.plot(y_g, list(zip(x1_g, x2_g)))
plt.title('At least on action (messages_count)')
plt.xlabel('Time mins')
plt.ylabel('Probability')
plt.show()
