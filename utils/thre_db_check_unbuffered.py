from threading import Thread
import os
from time import time


os.environ['DB_ENV'] = 'local'

from sqlalchemy import select, MetaData, event, alias, func
from edusson_ds_main.db.connections import DBConnectionsFacade
# from sqlalchemy.pool import QueuePool

# queue_pool = QueuePool()
# engine = DBConnectionsFacade.get_edusson_ds()
import sqlalchemy

engine = sqlalchemy.create_engine('mysql+pymysql://root@localhost/edusson_data_science',
                                  pool_recycle=500, pool_size=16, max_overflow=0)



meta = MetaData()
meta.reflect(bind=engine, only=['ds_metrics', 'ds_writer_metrics'])

DSMetrics = meta.tables['ds_metrics']
DSWriterMetrics = meta.tables['ds_writer_metrics']

class threads_object(Thread):
    def run(self):
        print('1')
        stime = time()

        # engine = sqlalchemy.create_engine('mysql+pymysql://root@localhost/edusson_data_science', pool_recycle=500, pool_size=16, max_overflow=0)
        q = DSWriterMetrics.select().where(DSWriterMetrics.c.metrics_id == 1).limit(10000)
        conn = engine.raw_connection()
        buffered_cursor = conn.cursor(buffered=False)

        # with engine.raw_connection() as connection:
        # buffered_cursor = buffered_cursor.raw_connection().cursor(buffered=False)
        buffered_cursor.execute(q, multi=True)

        # with buffered_cursor.connect() as connection:
        #     # res = connection.execute("SELECT ds_writer_metrics.id, ds_writer_metrics.writer_id, "
        #     #                          "ds_writer_metrics.metrics_id, ds_writer_metrics.metrics_value, "
        #     #                          "ds_writer_metrics.top_quantile, ds_writer_metrics.observation_count, "
        #     #                          "ds_writer_metrics.std_error, ds_writer_metrics.date_observation "
        #     #                          "FROM ds_writer_metrics "
        #     #                          "WHERE ds_writer_metrics.metrics_id = 3;").fetchall()
        #     res = connection.execute(q).fetchall()
        #     print('Res',len(res), time()-stime)



def threaded(num_iter):
    funcs = []
    for i in range(int(num_iter)):
        funcs.append(threads_object())
    for i in funcs:
        i.start()
    for i in funcs:
        i.join()


def show_results(func_name, results):
    print("%-23s %4.6f seconds" % (func_name, results))

if __name__ == '__main__':
    import sys

    number = 1
    num_threads = [1,2,4,8,16]

    threaded(1)

