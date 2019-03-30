import requests
from sqlalchemy import MetaData, select, insert
from sqlalchemy.engine import ResultProxy

base_url = 'https://www.gastronom.ru'
headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.9; rv:45.0) Gecko/20100101 Firefox/45.0'}
from bs4 import BeautifulSoup

import sqlalchemy
conn = sqlalchemy.create_engine('mysql+pymysql://root@localhost/alchemy?charset=utf8')
meta = MetaData()
meta.reflect(bind=conn)

Recipes = meta.tables['gastronom_recipes']
Ingredients = meta.tables['gastronom_ingredients']
Summary = meta.tables['gastronom_summary']

def check(url):
    query = select([Recipes]).where(Recipes.c.link == url).limit(1)
    return __exec(query).first()

def insert_receipt(title, url, preview, ingredients, summary, desc):
    ins = insert(Recipes).values(dict(
        name=title,
        link=url,
        preview=preview,
        description=".\n\n".join(desc),
    ))
    res = __exec(ins)
    inserted_id = res.inserted_primary_key[0]
    for key, val in summary:
        __exec(insert(Summary).values(dict(recipe_id=inserted_id, item=key, value=val)))

    for ingredient in ingredients:
        __exec(insert(Ingredients).values(dict(recipe_id=inserted_id, name=ingredient, quantity='')))


def __exec(*args, **kwargs) -> ResultProxy:
    with conn.connect() as c:
        return c.execute(*args, **kwargs)

def parse_page(url):
    if url[0] != '/':
        url = '/' + url

    if base_url not in url:
        url = base_url + url.strip()

    exists_url = check(url)
    print('url', url, exists_url.id if exists_url else 'None')

    if exists_url:
        return None

    try:
        r = requests.get(url, headers=headers, timeout=20)
    except Exception as e:
        print('Exception')
        print(e)
        return None

    soup = BeautifulSoup(r.text)

    title = ''
    preview = ''
    ingredients = ''
    summary = ''
    desc = ''

    title_soup = soup.find('h1', {'class': 'recipe__title'})

    preview_soup = soup.find('img', {'class':'result-photo'})
    ingredients_soup = soup.find_all('li', {'class':'recipe__ingredient'})

    list_titles_soup = soup.find_all('div', {'class': 'recipe__summary-list-title'})
    list_desc_soup = soup.find_all('div', {'class': 'recipe__summary-list-des'})

    desc_soup = soup.find_all('div', {'class': 'recipe__step-text'})

    if title_soup: title = title_soup.text.strip()
    if preview_soup: preview = base_url + preview_soup['src']
    if ingredients_soup: ingredients = [item.text for item in ingredients_soup]

    if list_titles_soup and list_desc_soup:
        list_titles = [item.text.strip() for item in list_titles_soup]
        list_desc = [item.text.strip() for item in list_desc_soup]
        summary = list(zip(list_titles, list_desc))

    if desc_soup: desc = [item.text.strip() for item in desc_soup]

    print(title, url, preview, ingredients, summary, desc)
    print('')


    try:
        insert_receipt(title, url, preview, ingredients, summary, desc)
    except Exception as e:
        print('Exception', e)
        for i in range(len(desc)):
            desc[i] = ''.join([w for w in desc[i] if ord(w) < 120000])
        title = ''.join([w for w in title if ord(w) < 10000])
        insert_receipt(title, url, preview, ingredients, summary, desc)


if __name__ == '__main__':
    with open('/home/andrei/Python/sqlalchemy-lab/receipt/links.txt', 'r') as f:
        for url in f.readlines():
            parse_page(url)

    # print(parse_page('/recipe/40263/zavtrak-s-lyubovyu'))