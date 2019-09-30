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
        function_to_run()


class non_threads_object(object):
    def run(self):
        function_to_run()


def non_threaded(num_iter):
    funcs = []
    for i in range(int(num_iter)):
        funcs.append(non_threads_object())
    for i in funcs:
        i.run()


def threaded(num_iter):
    funcs = []
    for i in range(int(num_iter)):
        funcs.append(threads_object())
    for i in funcs:
        i.start()
    for i in funcs:
        i.join()

def function_to_run():
    print('1')
    stime = time()

    # engine = sqlalchemy.create_engine('mysql+pymysql://root@localhost/edusson_data_science',
    #                                   pool_recycle=500, pool_size=16, max_overflow=0)
    q = DSWriterMetrics.select().where(DSWriterMetrics.c.metrics_id == 3)
    with engine.connect() as connection:
        res = connection.execute(q).fetchall()
        print('Res',len(res), time()-stime)


def show_results(func_name, results):
    print("%-23s %4.6f seconds" % (func_name, results))

if __name__ == '__main__':
    import sys
    from timeit import Timer

    repeat = 1
    number = 1
    num_threads = [1,2,4,8, 16]

print("start tersting")
for i in num_threads:
    print("============== start ======================")
    t = Timer("non_threaded(%s)" % i, "from __main__ import non_threaded")
    best_result = min(t.repeat(repeat=repeat, number=number))
    show_results("non threaded (%s iters)" % i, best_result)

    print("\n")

    t = Timer("threaded(%s)" % i, "from __main__ import threaded")
    best_result = min(t.repeat(repeat=repeat, number=number))
    show_results("threaded (%s iters)" % i, best_result)
    print("============== end ======================")
    print('')