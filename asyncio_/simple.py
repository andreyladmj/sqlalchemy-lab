import pandas as pd
import asyncio

import aiohttp
# import aiofiles

# f = aiofiles.open()

#aiofiles.open('filename', mode='r') as f:
from bs4 import BeautifulSoup

pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 2000)
pd.set_option('display.line_width', 2000)
file = open('/home/andrei/Python/sqlalchemy-lab/receipt/links.txt', 'r')

# def get_next_file_line():
#     yield from loop.run_in_executor(None, file.readline)

async def get_next_file_line():
    return await loop.run_in_executor(None, file.readline)

async def foo():
    print('Running in foo')
    await asyncio.sleep(0)
    print('Explicit context switch to foo again')

base_url = "https://www.gastronom.ru"
async def parse_web_page(url, session):
    if url[0] != '/':
        url = '/' + url

    url = base_url + url.strip()

    async with session.get(url, timeout=60) as response:
        res = await response.text()
        soup = BeautifulSoup(res)

        title = soup.find('h1', {'class': 'recipe__title'}).text
        preview = base_url + soup.find('img', {'class':'result-photo'})['src']
        ingredients = [item.text for item in soup.find_all('li', {'class':'recipe__ingredient'})]

        list_titles = [item.text.strip() for item in soup.find_all('div', {'class': 'recipe__summary-list-title'})]
        list_desc = [item.text.strip() for item in soup.find_all('div', {'class': 'recipe__summary-list-des'})]

        summary = list(zip(list_titles, list_desc))

        desc = [item.text.strip() for item in soup.find_all('div', {'class': 'recipe__step-text'})]

        return (title, url, preview, ingredients, summary, desc)


async def fetch_all_urls(session, loop):
    c = 1000
    n = 0

    data = [parse_web_page(await get_next_file_line(), session) for i in range(c)]

    while len(data) == c:
        res = await asyncio.gather(*data, return_exceptions=True)
        print('Iteration', n)
        print(res)
        n += 1
        process_data(res)
        data = []
        for i in range(c):
            url = await get_next_file_line()

            if not url:
                break

            data.append(parse_web_page(url, session))

    res = await asyncio.gather(*data, return_exceptions=True)
    process_data(res, True)
    return res
    # return await asyncio.gather(*[parse_web_page(url, session) for url in await get_next_file_line()], return_exceptions=True)

coocking_data = []
i = 1
def process_data(res, finish=False):
    global coocking_data, i

    coocking_data += res

    if len(coocking_data) >= 5000 or finish:
        df = pd.DataFrame(columns=['title', 'url', 'preview', 'ingredients', 'summary', 'desc'], data=coocking_data)
        df.to_pickle('/home/andrei/Python/sqlalchemy-lab/receipt/gastronom/receipts_{}.pkl'.format(i))
        i += 1
        coocking_data = []




loop = asyncio.get_event_loop()
connector = aiohttp.TCPConnector(limit=100)
session = aiohttp.ClientSession(loop=loop, connector=connector)
htmls = loop.run_until_complete(fetch_all_urls(session, loop))
loop.run_until_complete(session.close())
loop.close()
file.close()
