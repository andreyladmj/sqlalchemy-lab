import requests
from sqlalchemy import MetaData, select, insert
from sqlalchemy.engine import ResultProxy

base_url = 'https://povar.ru'
headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.9; rv:45.0) Gecko/20100101 Firefox/45.0'}
from bs4 import BeautifulSoup

import sqlalchemy
conn = sqlalchemy.create_engine('mysql+pymysql://root@localhost/alchemy?charset=utf8')
meta = MetaData()
meta.reflect(bind=conn)

Recipes = meta.tables['povar_recipes']
Ingredients = meta.tables['povar_ingredients']
Summary = meta.tables['povar_summary']

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

    title_soup = soup.find('h1', {'class': 'detailed'})

    preview_soup = soup.find('div', {'class':'bigImgBox'})
    if preview_soup:
        preview_soup = preview_soup.find('img')

    ingredients_soup = soup.find_all('li', {'itemprop':'ingredients'})

    detailed_tags_soup = soup.find('span', {'class':'detailed_tags'})
    # if detailed_tags_soup:
    #     list_titles_soup = detailed_tags_soup.find_all('span', {'class': 'b'})
    #     list_desc_soup = detailed_tags_soup.find_all('a')

    summary_titles = []
    summary_values = []
    tmp_vals = []
    start = False
    if detailed_tags_soup:
        for item in detailed_tags_soup.children:
            if str(item).strip():
                if start:
                    if item.name == 'span' and 'b' in item['class']:
                        summary_titles.append(item.text)
                    if item.name == 'a':
                        tmp_vals.append(item.text)

                    if item.name == 'br':
                        summary_values.append(' / '.join(tmp_vals))
                        tmp_vals = []

                if item.name == 'br':
                    start = True

    summary_values.append(' / '.join(tmp_vals))

    desc_soup = soup.find_all('div', {'class': 'detailed_step_description_big'})

    if title_soup: title = title_soup.text.strip()
    if preview_soup: preview = base_url + preview_soup['src']
    if ingredients_soup: ingredients = [item.text.replace('\xa0', '').replace('                            ', '').strip() for item in ingredients_soup]

    if summary_titles:
        # list_titles = [item.text.strip() for item in list_titles_soup]
        # list_desc = [item.text.strip() for item in list_desc_soup]
        summary = list(zip(summary_titles, summary_values))

    if desc_soup: desc = [item.text.strip() for item in desc_soup]

    print(title, url, preview, ingredients, summary, desc)
    # print(url, ingredients)
    print('')
    insert_receipt(title, url, preview, ingredients, summary, desc)


if __name__ == '__main__':
    with open('/home/andrei/Python/sqlalchemy-lab/receipt/povar_links.txt', 'r') as f:
        for url in f.readlines():
            parse_page(url)

    # print(check('https://www.gastronom.ru/recipe/28892/video-recipe-adzhapsandali'))