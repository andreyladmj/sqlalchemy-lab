import asyncio

import aiohttp
# import aiofiles

# f = aiofiles.open()

#aiofiles.open('filename', mode='r') as f:

file = open('/home/andrei/Python/sqlalchemy-lab/receipt/catalog.txt', 'r')

# def get_next_file_line():
#     yield from loop.run_in_executor(None, file.readline)

async def get_next_file_line():
    return await loop.run_in_executor(None, file.readline)

async def foo():
    print('Running in foo')
    await asyncio.sleep(0)
    print('Explicit context switch to foo again')


async def parse_web_page(url, session):
    print('parse_web_page', url)
    async with session.get("https://www.gastronom.ru"+url, timeout=60) as response:
        res = await response.text()
        print('parse_web_page', url, 'got', len(res))
        return len(res)


async def fetch_all_urls(session, loop):
    return await asyncio.gather(*[parse_web_page(url, session) for url in await get_next_file_line()], return_exceptions=True)


urls = [
    'https://povar.ru/recipes/manty_po-uzbekski-18397.html',
    'https://povar.ru/recipes/litovskie_ceppeliny-8617.html',
    'https://povar.ru/recipes/podliv_iz_svininy-7633.html',
]

loop = asyncio.get_event_loop()
connector = aiohttp.TCPConnector(limit=100)
session = aiohttp.ClientSession(loop=loop, connector=connector)
htmls = loop.run_until_complete(fetch_all_urls(session, loop))
loop.run_until_complete(session.close())
loop.close()
file.close()

print(htmls)