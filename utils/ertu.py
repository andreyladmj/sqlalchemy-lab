from datetime import datetime, timedelta

import pandas as pd
import os
os.environ['DB_ENV'] = 'prod'
from edusson_ds_main.db.connections import DBConnectionsFacade

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 2000)

sql2 = """

select t2.order_id from es_task t2
LEFT join es_task_support_sale t1 on t1.id=t2.id
where t1.support_sale_proof_id in (1,4,2) 
"""

sql = """
SELECT *
FROM es_orders o 
LEFT JOIN es_product p1 ON p1.order_id = o.order_id
LEFT JOIN es_product_type_essay pe1 ON p1.product_id = pe1.product_id
where o.test_order = 0 
and o.is_first_client_order = 1

and o.order_id not in (
    select distinct p.order_id from es_order_preset p where p.order_id is not null
)

"""
df_exclude = pd.read_sql(sql2, con=DBConnectionsFacade.get_edusson_replica())
df = pd.read_sql(sql, con=DBConnectionsFacade.get_edusson_replica())

df.is_paid_order.describe()

stored_df = df.copy()
# df = df[['site_id', 'order_id', 'order_total_usd', 'pages', 'deadline', 'order_date', 'is_first_client_order', 'is_easy_bidding']].drop_duplicates()[~df.order_id.isin(df_exclude.order_id.values)]
#df = df.groupby(df.index).first()

cols = []
count = 1
for column in df.columns:
    if column == 'order_id':
        cols.append('order_id'+str(count))
        count+=1
        continue
    cols.append(column)
df.columns = cols

df = df.rename(columns={'order_id1':'order_id'})

df = df[~df.order_id.isin(df_exclude.order_id.values)]


'''
SELECT
 #sum(t0.order_total_usd)/count(t0.order_id) as k1,
 t0.order_id,
 t0.estimated_total
 #case when t1.support_sale_proof_id=1 then count(t0.order_id) end as t1,
 #case when t1.support_sale_proof_id=4 then count(t0.order_id) end as t4,
#case when t1.support_sale_proof_id=2 then count(t0.order_id) end as t2
 from es_orders as t0
   join es_task t2 on t2.order_id=t0.order_id
   join es_task_support_sale t1 on t1.id=t2.id
 where t0.test_order = 0
       and t0.site_id != 31
       and t0.site_id = 8
       and t0.order_date between '2019-07-01 00:00:00' and '2019-12-31 23:59:59'
       and t0.is_paid_order=1
       and t0.is_first_client_order=1
       and t2.state_id=7
       and t1.support_sale_proof_id in (1,4,2)
'''

# stored_df = df.copy()

df['price_per_page'] = df.order_total_usd / df.pages

# df = df[df.test_order == 0]

df['date_diff'] = df.deadline - df.order_date

dtime = timedelta(hours=6)

start_datetime = datetime.strptime("2019-01-01", "%Y-%m-%d")

# df = stored_df.copy()

exclude = [30, 31, 32, 35, 49, 50]
include = [30, 32, 35, 49, 50]

df = df[df.order_date > start_datetime]
df = df[df.is_first_client_order == 1]
df = df[~df.site_id.isin(exclude)]
# df = df[df.site_id.isin(include)]

short_df = df[df.date_diff < dtime]
long_df = df[df.date_diff >= dtime]

# print(f'Data from (order date > {start_datetime}), only for FCO')
# print(f'Sites ids: {",".join(df.site_id.unique().astype(str))}')
# print(f'Short orders (time less than {dtime}) p2p is: {short_df.is_paid_order.mean()}, ({len(short_df)} orders)')
# print(f'Orders total usd mean between paid orders: {short_df[short_df.is_paid_order == 1].order_total_usd.mean()}')
# print(f'Mean price per page: {short_df[short_df.is_paid_order == 1].price_per_page.mean()}')
# print('')
# print(f'Long orders (time more than or equal {dtime}) p2p is: {long_df.is_paid_order.mean()}, ({len(long_df)} orders)')
# print(f'Orders total usd mean between paid orders: {long_df[long_df.is_paid_order == 1].order_total_usd.mean()}')
# print(f'Mean price per page: {long_df[long_df.is_paid_order == 1].price_per_page.mean()}')


short_easy_bidding = short_df[short_df.is_easy_bidding == 1]
short_not_easy_bidding = short_df[short_df.is_easy_bidding == 0]

long_easy_bidding = long_df[long_df.is_easy_bidding == 1]
long_not_easy_bidding = long_df[long_df.is_easy_bidding == 0]



print(f"Total number of orders: {len(df)}")
print(f'Data from (order date > {start_datetime}), only for FCO')
print(f'Sites ids: {",".join(df.site_id.unique().astype(str))}')
print('')
print(f'Short EASY BIDDING orders (time less than {dtime}) p2p is: {short_easy_bidding.is_paid_order.mean()}, ({len(short_easy_bidding)} orders)')
print(f'Orders total usd mean between paid orders: {short_easy_bidding[short_easy_bidding.is_paid_order == 1].order_total_usd.mean()}')
print(f'Mean price per page: {short_easy_bidding[short_easy_bidding.is_paid_order == 1].price_per_page.mean()}')
print('')
print(f'Short NOT EASY BIDDING orders (time less than {dtime}) p2p is: {short_not_easy_bidding.is_paid_order.mean()}, ({len(short_not_easy_bidding)} orders)')
print(f'Orders total usd mean between paid orders: {short_not_easy_bidding[short_not_easy_bidding.is_paid_order == 1].order_total_usd.mean()}')
print(f'Mean price per page: {short_not_easy_bidding[short_not_easy_bidding.is_paid_order == 1].price_per_page.mean()}')
print('')
print('')
print(f'Long EASY BIDDING orders (time more than or equal {dtime}) p2p is: {long_easy_bidding.is_paid_order.mean()}, ({len(long_easy_bidding)} orders)')
print(f'Orders total usd mean between paid orders: {long_easy_bidding[long_easy_bidding.is_paid_order == 1].order_total_usd.mean()}')
print(f'Mean price per page: {long_easy_bidding[long_easy_bidding.is_paid_order == 1].price_per_page.mean()}')
print('')
print(f'Long NOT EASY BIDDING orders (time more than or equal {dtime}) p2p is: {long_not_easy_bidding.is_paid_order.mean()}, ({len(long_not_easy_bidding)} orders)')
print(f'Orders total usd mean between paid orders: {long_not_easy_bidding[long_not_easy_bidding.is_paid_order == 1].order_total_usd.mean()}')
print(f'Mean price per page: {long_not_easy_bidding[long_not_easy_bidding.is_paid_order == 1].price_per_page.mean()}')

#
# # short_df.is_paid_order.mean()
# # short_df[short_df.is_paid_order == 1].order_total_usd.mean()
# #
# # long_df.is_paid_order.mean()
# # long_df[long_df.is_paid_order == 1].order_total_usd.mean()
#
# import numpy as np
# p = 0.04
# n = 20
# z = 2
#
# print(f'p: {p}, n: {n}, z: {z}')
# print('from', (2*n*p + z**2 - (2*np.sqrt(z**2 - 1/n + 4*n*p*(1-p) + (4*p-2))+1)) / (2*(n+z**2)))
# print('to', (2*n*p + z**2 + (2*np.sqrt(z**2 - 1/n + 4*n*p*(1-p) - (4*p-2))+1)) / (2*(n+z**2)))
# print('')
#
# p = 0.04
# n = 2
# z = 2
#
# print(f'p: {p}, n: {n}, z: {z}')
# print('from', (2*n*p + z**2 - (2*np.sqrt(z**2 - 1/n + 4*n*p*(1-p) + (4*p-2))+1)) / (2*(n+z**2)))
# print('to', (2*n*p + z**2 + (2*np.sqrt(z**2 - 1/n + 4*n*p*(1-p) - (4*p-2))+1)) / (2*(n+z**2)))
# print('')
#
#
# p = 0.98
# n = 20
# z = 2
#
# print(f'p: {p}, n: {n}, z: {z}')
# print('from', (2*n*p + z**2 - (2*np.sqrt(z**2 - 1/n + 4*n*p*(1-p) + (4*p-2))+1)) / (2*(n+z**2)))
# print('to', (2*n*p + z**2 + (2*np.sqrt(z**2 - 1/n + 4*n*p*(1-p) - (4*p-2))+1)) / (2*(n+z**2)))
