import multiprocessing as mp
import pandas as pd
import numpy as np
import ctypes

df = pd.DataFrame([
    [1,11,111,1111,1111],
    [2,22,222,2222,2222],
    [3,33,333,3333,3333],
    [4,44,444,4444,4444],
    [5,55,555,5555,5555],
    [6,66,666,6666,6666],
    [7,77,777,7777,7777]
])

df_rows = df.shape[0]
df_cols = df.shape[1]

res_mean = mp.Array(ctypes.c_double, df_rows)
res_square = mp.Array(ctypes.c_double, df_rows * df_cols)

t = np.array_split(df, 3)[0]

res_mean[:3] = t.mean(axis=1)


def calc_mean(i, df, res_mean, res_square):
    l = len(df)
    res_mean[i*l]


def calc_mean_squares(df):
    return df.mean(axis=1)
    # return df.mean(axis=1), df**2

if __name__ == '__main__':
    pn = 3
    with mp.Pool(pn) as p:
        res = p.map(calc_mean_squares, np.array_split(df, pn))
    pd.concat(res)


if __name__ == '__main__2':
    pn = 3
    procs = []

    for i, df_batch in enumerate(np.array_split(df, pn)):
        p = mp.Process(target=calc_mean, args=(i, df_batch, res_mean, res_square))
        procs.append(p)

    for p in procs:
        p.start()

    for p in procs:
        p.join()

        print(df)
        print(res_mean)
        print(res_square)