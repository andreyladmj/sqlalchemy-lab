import pandas as pd

import sqlalchemy
from sqlalchemy import MetaData, event

conn = sqlalchemy.create_engine('mysql+pymysql://root@localhost/alchemy?charset=utf8')

meta = MetaData()
meta.reflect(bind=conn)

AllIngredients = meta.tables['all_ingredients']

import numpy as np

def add_own_encoders(conn, cursor, query, *args):
    cursor.connection.encoders[np.int64] = lambda value, encoders: int(value)
    cursor.connection.encoders[np.float64] = lambda value, encoders: float(value)
    cursor.connection.encoders[pd.Timestamp] = lambda value, encoders: encoders[str](str(value.to_pydatetime()))
    cursor.connection.encoders[pd.Timedelta] = lambda value, encoders: value.total_seconds()

event.listen(conn, "before_cursor_execute", add_own_encoders)


pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 2000)

df_receipts = pd.read_sql("SELECT * FROM receipts;", conn, index_col='id')
df_ingredients = pd.read_sql("SELECT * FROM ingredients;", conn)

df_ingredients = pd.concat([pd.read_pickle('receipt/df_ingredients_part_{}.pkl'.format(i)) for i in [0,1,2]])
df_receipts = pd.concat([pd.read_pickle('receipt/df_receipts_part_{}.pkl'.format(i)) for i in [0,1,2,3,4,5]])

# for i, part in enumerate(np.array_split(df_receipts, 6)):
#     part.to_pickle('receipt/df_receipts_part_{}.pkl'.format(i))
# for i, part in enumerate(np.array_split(df_ingredients, 3)):
#     part.to_pickle('receipt/df_ingredients_part_{}.pkl'.format(i))

df = df_ingredients.join(df_receipts, on='receipt_id', lsuffix='_ing').set_index('id')
df.type.unique()
df[df.type == 'ivona'].name_ing.unique()
# df.loc[df.type == 'ivona', 'full_name'] = df[df.type == 'ivona'].name_ing


df[df.type == 'ivona'].groupby('name_ing').name_ing.count().sort_values()

# others = df[df.type != 'ivona'].name_ing.apply(lambda x: x.split(' – ')[0]).apply(lambda x: x.split('—')[0])
others = df.name_ing.apply(lambda x: x.split(' – ')[0]).apply(lambda x: x.split('—')[0].strip().lower())
others.to_frame('name_ing').groupby('name_ing').name_ing.count().sort_values()