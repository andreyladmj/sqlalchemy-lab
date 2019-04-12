import pandas as pd
import numpy as np

import sqlalchemy
from sqlalchemy import MetaData, event, select

conn = sqlalchemy.create_engine('mysql+pymysql://root@localhost/alchemy?charset=utf8')

meta = MetaData()
meta.reflect(bind=conn)

gastronom_ingredients = meta.tables['gastronom_ingredients']
gastronom_recipes = meta.tables['gastronom_recipes']

parsed_recipes = meta.tables['parsed_recipes']
parsed_ingredients = meta.tables['parsed_ingredients']

povar_recipes = meta.tables['povar_recipes']
povar_ingredients = meta.tables['povar_ingredients']

receipts = meta.tables['receipts']
ingredients = meta.tables['ingredients']


def exec_(query):
    with conn.connect() as c:
        return c.execute(query)

def move_receipts():
    g_receipts = exec_(select([gastronom_recipes])).fetchall()
    for receipt in g_receipts:
        q = receipts.insert().values({
            'name': receipt.name,
            'preview': receipt.preview,
            'link': receipt.link,
            'description': receipt.description,
            'cooking_time': 0,
            'portions': 0,
            'difficult': 0,
            'public': 0,
            'created_at': receipt.created_at,
            'original_id': receipt.id,
            'type': 'gastronom',
        })
        exec_(q)

    p_receipts = exec_(select([parsed_recipes])).fetchall()
    for receipt in p_receipts:
        q = receipts.insert().values({
            'name': receipt.name,
            'preview': receipt.preview,
            'link': receipt.link,
            'description': receipt.description,
            'cooking_time': 0,
            'portions': 0,
            'difficult': 0,
            'public': 0,
            'created_at': receipt.created_at,
            'original_id': receipt.id,
            'type': 'ivona',
        })
        exec_(q)

    povar_receipts_all = exec_(select([povar_recipes])).fetchall()
    for receipt in povar_receipts_all:
        q = receipts.insert().values({
            'name': receipt.name,
            'preview': receipt.preview,
            'link': receipt.link,
            'description': receipt.description,
            'cooking_time': 0,
            'portions': 0,
            'difficult': 0,
            'public': 0,
            'created_at': receipt.created_at,
            'original_id': receipt.id,
            'type': 'povar',
        })
        exec_(q)


def move_ingredients():
    gastronom_ing_all = exec_(select([gastronom_ingredients])).fetchall()
    for ing in gastronom_ing_all:
        q = ingredients.insert().values({
            'name': ing.name,
            'created_at': ing.created_at,
            'receipt_id': select([receipts.c.id]).where(receipts.c.original_id == ing.recipe_id).where(receipts.c.type == 'gastronom'),
            'full_name': ing.name_full,
            'quantity': ing.quantity,
        })
        exec_(q)

    ivona_ing_all = exec_(select([parsed_ingredients])).fetchall()
    for ing in ivona_ing_all:
        q = ingredients.insert().values({
            'name': ing.name,
            'created_at': ing.created_at,
            'receipt_id': select([receipts.c.id]).where(receipts.c.original_id == ing.parsed_recipe_id).where(receipts.c.type == 'ivona'),
            'full_name': ing.name + ' ' + str(ing.quantity),
            'quantity': ing.quantity,
        })
        exec_(q)

    povar_ing_all = exec_(select([povar_ingredients])).fetchall()
    for ing in povar_ing_all:
        q = ingredients.insert().values({
            'name': ing.name,
            'created_at': ing.created_at,
            'receipt_id': select([receipts.c.id]).where(receipts.c.original_id == ing.recipe_id).where(receipts.c.type == 'povar'),
            'full_name': ing.name,
            'quantity': ing.quantity,
        })
        exec_(q)

