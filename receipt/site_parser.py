import requests
#base_url = 'https://www.gastronom.ru'
headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.9; rv:45.0) Gecko/20100101 Firefox/45.0'}
from bs4 import BeautifulSoup
from lxml import html

sites = [
    ('https://povar.ru', '/list'),
    ('https://www.gastronom.ru', '/catalog'),
]

class SiteParser:
    domain = None
    start_catalog = None

    def __init__(self):
        self.base_content = self.get_content_soup(self.start_catalog)

    def get_content_soup(self, url):
        if self.domain not in url:
            url = self.domain + url
        try:
            r = requests.get(url, headers=headers, timeout=10)
        except Exception as e:
            print('Exception', e)
            return None

        return BeautifulSoup(r.text)

    def get_categories_links(self):
        pass


class PovarParser(SiteParser):
    domain = 'https://povar.ru'
    start_catalog = 'https://povar.ru/list'

    def get_categories_links(self):
        return self.base_content.find_all('a', {'class', 'itemHdr'})

    def parse_page(self, url, page):
        if page == 1:
            page = ''

        url = url + str(page)
        soup = self.get_content_soup(url)

        if soup:
            return soup.find_all('a', {'class': 'listRecipieTitle'})

        return []


povar = PovarParser()
categories = povar.get_categories_links()

all_links = []
checked_categories = []

# povar.parse_page(categories[0]['href'], 900000) tests

for i, category_url in enumerate(categories):
    for page in range(10000):
        links = povar.parse_page(category_url['href'], page + 1)
        print('Parsed', category_url['href'], i, 'from', len(categories), 'page', page + 1, 'got', len(links))
        all_links += links

        if not len(links):
            break

    checked_categories.append(category_url['href'])
    with open('receipt/povar_catalog.txt', 'a') as f:
        f.write(category_url['href']+'\n')

    with open('receipt/povar_links.txt', 'a') as f:
        for link in all_links:
            f.write(link['href']+'\n')
        all_links = []