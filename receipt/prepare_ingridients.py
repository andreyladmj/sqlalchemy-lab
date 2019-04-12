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

df = pd.read_sql("SELECT * FROM gastronom_ingredients", conn)

df.name_full = df.name_full.apply(lambda x: x.replace('–', '-'))

def split_ingredient_string(row):
    args = row.split('-', 1)
    if len(args) > 1:
        if args[0].isdigit() or args[0].replace(',', '.').isdigit():
            return [row]

        args[0] = args[0].strip()
        args[1] = args[1].strip()

    return args

df_parts = df.name_full.apply(split_ingredient_string).to_frame()
df_parts['part_counts'] = df_parts.name_full.apply(len)
df_parts.part_counts.value_counts()



_parts_df_2 = df_parts[df_parts.part_counts == 2]
_parts_df_2['first_arg'] = _parts_df_2.name_full.apply(lambda x: x[0].strip())
all_ingredients = _parts_df_2.first_arg.value_counts().to_frame()
all_ingredients[all_ingredients.first_arg < 10]



_parts_df_1 = df_parts[df_parts.part_counts == 1]
_parts_df_1['first_arg'] = _parts_df_1.name_full.apply(lambda x: x[0].strip())
all_ingredients = _parts_df_1.first_arg.value_counts().to_frame()
all_ingredients[all_ingredients.first_arg < 10]

with conn.connect() as c:
    for name, v in all_ingredients.iterrows():
        value = v.values[0]
        print(name, value)
        q = AllIngredients.insert().values({'name': name, 'value_count': value})
        c.execute(q)
        # c.execute('INSERT INTO all_ingredients VALUES (NULL, "{}", {});'.format(name, value))

ingredients = []

with open('/home/andrei/Python/sqlalchemy-lab/receipt/ingredientsfromsite.txt', 'r') as f:
    with conn.connect() as c:
        for item in f.readlines():
            item = item.strip()

            if not item or len(item) < 3:
                continue

            ingredients.append(item.lower())

def get_jaccard_sim(str1, str2):
    a = set(str1.split())
    b = set(str2.split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))



import nltk
nltk.edit_distance("humpty", "dumpty")
nltk.edit_distance('вино', '50 г черного винограда')
nltk.edit_distance('виноград', '50 г черного винограда')

tdf['edit_distance'] = tdf.name_full.apply(lambda name: [nltk.edit_distance(item, name) for item in ingredients])

tdf['first_5'] = tdf.edit_distance.apply(lambda x: np.array(x).argsort()[:5])
tdf['items_5'] = tdf.first_5.apply(lambda x: [ingredients[i] for i in x])

tdf[['name_full', 'items_5']]



additional_ings = pd.read_sql('SELECT DISTINCT name FROM parsed_ingredients WHERE quantity != "";', conn)
for i in additional_ings.values:
    if i not in ingredients:
        print(i)



additional_ings = pd.read_sql('SELECT DISTINCT name FROM parsed_ingredients WHERE quantity != "";', conn)