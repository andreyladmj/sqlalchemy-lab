from concurrent import futures
import os
from time import time

import pandas as pd

folder = '/home/andrei/Python/data_science/order-form/test_model_with_min_words_count'


def foo(row):
    print(os.getpid(), row, columns)
    print(row[columns.index('tokenized_description')])
    print('')
    return os.getpid()

df = pd.read_pickle('{}/tokenized.pkl'.format(folder))
columns = list(df.columns)
df = df[:10]
most_popular_names_words = pd.read_pickle('{}/most_popular_names_words.pkl'.format(folder))[0].values
most_popular_desc_words = pd.read_pickle('{}/most_popular_desc_words.pkl'.format(folder))[0].values

stime = time()
with futures.ProcessPoolExecutor() as pool:
    res = pool.map(foo, df.values)
    print(list(res))

print('End', time()-stime)



stime = time()
with futures.ProcessPoolExecutor() as pool:
    # running = [pool.submit(foo, 8, 7), pool.submit(foo, 8, 7)]
    running = pool.submit(map(foo, df.values))

    for future in futures.as_completed(running):
        data = future.result()
print('End', time()-stime)