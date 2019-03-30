from datetime import datetime, timedelta
from time import time

import pandas as pd
import sqlalchemy
import os
os.environ['DB_ENV'] = 'prod'
from dateutil.relativedelta import *
from edusson_ds_main.db.connections import DBConnectionsFacade

sql = """
SELECT
    order_id,
    rev,
    order_date,
    deadline,
    estimated_total,
    name,
    description
FROM es_orders_audit
WHERE order_total_usd = 0
AND order_date BETWEEN "{}" AND "{}"
AND test_order = 0
AND site_id != 31
GROUP BY order_id, order_date;
"""

start_date = datetime(year=2016, month=1, day=1)


def save_audit_data_by_month(start_date):
    to_date = (start_date + relativedelta(months=1)).strftime('%Y-%m-%d')
    from_date = start_date.strftime('%Y-%m-%d')
    stime = time()
    df = pd.read_sql(sql=sql.format(from_date, to_date), con=DBConnectionsFacade.get_edusson_replica())
    df.to_pickle('/home/andrei/Python/sqlalchemy-lab/p2paid/audit_data2/{}-{}.{}.pkl'.format(from_date, to_date, len(df)))
    print("From {} to {} len: {}, time: {}".format(from_date, to_date, len(df), time()-stime))


# while start_date < datetime.now():
#     save_audit_data_by_month(start_date)
#     start_date = start_date + relativedelta(months=1)

from os import listdir
from os.path import isfile, join
mypath = '/home/andrei/Python/sqlalchemy-lab/p2paid/audit_data'
audit_data_files = [f for f in listdir(mypath) if isfile(join(mypath, f))]

with DBConnectionsFacade().get_edusson_ds().connect() as conn:
    conn.execute('DROP TABLE IF EXISTS es_orders_audit;')

audit_data_files = sorted(audit_data_files)

for file in audit_data_files:
    df = pd.read_pickle('{}/{}'.format(mypath, file))
    df = df.set_index(['order_id', 'order_date'])
    df.to_sql('es_orders_audit', con=DBConnectionsFacade().get_edusson_ds(), if_exists='append')
    print('Processed', file)
