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

# for i, part in enumerate(np.array_split(df_receipts, 6)):
#     part.to_pickle('receipt/df_receipts_part_{}.pkl'.format(i))
# for i, part in enumerate(np.array_split(df_ingredients, 3)):
#     part.to_pickle('receipt/df_ingredients_part_{}.pkl'.format(i))

df = df_ingredients.join(df_receipts, on='receipt_id', lsuffix='_ing').set_index('id')
df.type.unique()
# df[df.type == 'ivona'].name_ing.unique()
#
# df[df.type == 'ivona'].groupby('name_ing').name_ing.count().sort_values()
#
# others = df[df.type != 'ivona'].name_ing.apply(lambda x: x.split(' – ')[0]).apply(lambda x: x.split('—')[0])
df.loc[df.type != 'ivona', 'name_ing'] = df[df.type != 'ivona'].name_ing.apply(lambda x: x.split(' – ')[0]).apply(lambda x: x.split('—')[0])

df.name_ing = df.name_ing.apply(lambda x: x.lower())

uniq_ings = df.groupby('name_ing').name_ing.count().sort_values()

uniq_ings = uniq_ings.to_frame('count').reset_index().rename(columns={'name_ing':'name'})
uniq_ings.index.name = 'id'

# df = df[:10000]
#
# i = 0
# def trr(x):
#     global i
#     v = uniq_ings[uniq_ings.name == x].index
#     print(i, x, v)
#     i += 1
#     return v
#
# df['ing_index'] = df.name_ing.apply(trr)






# uu = df.join(uniq_ings, on='name')

uniq_ings['ing'] = ''
uniq_ings.loc[uniq_ings.name.str.contains("лук"), 'ing'] = 'лук'
uniq_ings.loc[uniq_ings.name.str.contains("яйц"), 'ing'] = 'яйцо'
uniq_ings.loc[(uniq_ings.name.str.contains("кунжут")) & (uniq_ings.name.str.contains("масл")), 'ing'] = 'кунжутное масло'
uniq_ings.loc[(uniq_ings.name.str.contains("оливков")) & (uniq_ings.name.str.contains("масл")), 'ing'] = 'оливковое масло'
uniq_ings.loc[(uniq_ings.name.str.contains("растит")) & (uniq_ings.name.str.contains("масл")), 'ing'] = 'растительное масло'
uniq_ings.loc[(uniq_ings.name.str.contains("сливоч")) & (uniq_ings.name.str.contains("масл")), 'ing'] = 'сливочное масло'
uniq_ings.loc[(uniq_ings.name.str.contains("подсолн")) & (uniq_ings.name.str.contains("масл")), 'ing'] = 'подсолнечное масло'
uniq_ings.loc[uniq_ings.name.str.contains("молок"), 'ing'] = 'молоко'
uniq_ings.loc[(uniq_ings.name.str.contains("сгущен")) & (uniq_ings.name.str.contains("молок")), 'ing'] = 'cгущённое молоко'
uniq_ings.loc[(uniq_ings.name.str.contains("сгущён")) & (uniq_ings.name.str.contains("молок")), 'ing'] = 'cгущённое молоко'
uniq_ings.loc[uniq_ings.name.str.contains("сметан"), 'ing'] = 'сметана'
uniq_ings.loc[uniq_ings.name.str.contains("сахар"), 'ing'] = 'сахар'
uniq_ings.loc[(uniq_ings.name.str.contains("ваниль")) & (uniq_ings.name.str.contains("сахар")), 'ing'] = 'ванильный сахар'
uniq_ings.loc[uniq_ings.name.str.contains("картоф"), 'ing'] = 'картофель'
uniq_ings.loc[uniq_ings.name.str.contains("морковь"), 'ing'] = 'морковь'
uniq_ings.loc[uniq_ings.name.str.contains("мука"), 'ing'] = 'мука'
uniq_ings.loc[uniq_ings.name.str.contains("лимон"), 'ing'] = 'лимон'
uniq_ings.loc[(uniq_ings.name.str.contains("сок")) & (uniq_ings.name.str.contains("лимон")), 'ing'] = 'лимонный сок'

uniq_ings[uniq_ings.name.str.contains("рис")]
# uniq_ings[uniq_ings.name.str.contains("майонез")]

uniq_ings.to_sql('unique_ingredients', con=conn, if_exists='append')
