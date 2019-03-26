import requests
base_url = 'https://www.gastronom.ru'
headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.9; rv:45.0) Gecko/20100101 Firefox/45.0'}
# r = requests.get(base_url + '/catalog', headers=headers)
#
# text = r.text.encode('cp1251')
#

from bs4 import BeautifulSoup
from lxml import html
#
# # Beautiful Soup
# soup = BeautifulSoup(text)
# catalogs = soup.find_all('a', {'class': 'col-catalog__link_cloud'})
#
# for catalog in catalogs:
#     url = base_url + catalog['href']
#     catalog_response = requests.get(url, headers=headers)
#     catalog_soup = BeautifulSoup(catalog_response.text.encode('cp1251'))
#     all_links = catalog_soup.find_all('a', {'class': 'material-anons__title'})
#
#     with open('links.txt', 'a') as f:
#         for link in all_links:
#             f.write(base_url + link['href'])



# sites = [
#     ('https://povar.ru', '/list'),
#     ('https://www.gastronom.ru', '/catalog'),
# ]
#
# class SiteParser:
#     def __init__(self, domain, start_catalog):
#         self.domain = domain
#         self.start_catalog = start_catalog
#         self.base_content = self.get_content_soup(self.start_catalog, self.domain)
#
#     def get_content_soup(self, url, base_url):
#         if base_url not in url:
#             url = base_url + url
#         r = requests.get(url, headers=headers)
#         return BeautifulSoup(r.text)
#
#     def get_categories_links(self):
#         pass



def get_links(url, class_=''):
    if base_url not in url:
        url = base_url + url
    r = requests.get(url, headers=headers)
    soup = BeautifulSoup(r.text.encode('cp1251'))
    return soup.find_all('a', {'class': class_})


def get_receipt_links(url, class_=''):
    if base_url not in url:
        url = base_url + url
    try:
        r = requests.get(url, headers=headers, timeout=10)
    except:
        return []
    soup = BeautifulSoup(r.text)
    div_s = soup.find('div', {'class', 'archive'})
    if div_s:
        return div_s.find_all('a', {'class': class_})
    return []

catalogs = get_links('/catalog', 'col-catalog__link_cloud')
all_links = []
checked_catalogs = []

for i, catalog in enumerate(catalogs):
    for page in range(1000):
        links = get_receipt_links(catalog['href'] + '?page={}'.format(page), 'material-anons__title')
        print('Parse', catalog, i, 'from', len(catalogs), 'page', page, 'got', len(links))
        all_links += links

        if not len(links):
            break

    checked_catalogs.append(catalog['href'])
    with open('receipt/catalog.txt', 'a') as f:
        f.write(catalog['href']+'\n')

    with open('receipt/links.txt', 'a') as f:
        for link in all_links:
            f.write(link['href']+'\n')
        all_links = []

# # lxml
# tree = html.fromstring(text)
# film_list_lxml = tree.xpath('//div[@class = "profileFilmsList"]')[0]