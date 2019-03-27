import asyncio
import logging

import aiohttp
import requests

headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.9; rv:45.0) Gecko/20100101 Firefox/45.0'}

logger = logging.getLogger('asyncio')
logger.setLevel(logging.DEBUG)

async def fetch_url(session, url):
    async with session.get(url, timeout=60) as response:
        return await response.text()



async def fetch_all_urls(session, urls, loop):
    logger.info('fetch_url')
    results = await asyncio.gather(*[fetch_url(session, url) for url in urls],
                                   return_exceptions=True)
    return results


def get_htmls(urls):
    if len(urls) > 1:
        loop = asyncio.get_event_loop()
        connector = aiohttp.TCPConnector(limit=100)
        with aiohttp.ClientSession(loop=loop, connector=connector) as session:
            htmls = loop.run_until_complete(fetch_all_urls(session, urls, loop))
        raw_result = dict(zip(urls, htmls))
    else:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
        raw_result = requests.get(urls[0], headers=headers).text

    return raw_result


url_list = [
    'https://povar.ru/recipes/manty_po-uzbekski-18397.html',
    'https://povar.ru/recipes/litovskie_ceppeliny-8617.html',
    'https://povar.ru/recipes/podliv_iz_svininy-7633.html'
]
result_dict = get_htmls(url_list)
print('result_dict')
print(len(result_dict))
