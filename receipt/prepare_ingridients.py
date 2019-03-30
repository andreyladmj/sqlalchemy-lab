import pandas as pd

import sqlalchemy
conn = sqlalchemy.create_engine('mysql+pymysql://root@localhost/alchemy?charset=utf8')

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