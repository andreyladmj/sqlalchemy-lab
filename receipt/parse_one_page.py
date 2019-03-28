import requests
base_url = 'https://www.gastronom.ru'
headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.9; rv:45.0) Gecko/20100101 Firefox/45.0'}
r = requests.get('https://www.gastronom.ru/recipe/43087/sochniki-s-tvorogom-i-vishnej', headers=headers)
from bs4 import BeautifulSoup

soup = BeautifulSoup(r.text)


title = soup.find('h1', {'class': 'recipe__title'}).text
preview = base_url + soup.find('img', {'class':'result-photo'})['src']
ingredients = [item.text for item in soup.find_all('li', {'class':'recipe__ingredient'})]

list_titles = [item.text.strip() for item in soup.find_all('div', {'class': 'recipe__summary-list-title'})]
list_desc = [item.text.strip() for item in soup.find_all('div', {'class': 'recipe__summary-list-des'})]

summary = list(zip(list_titles, list_desc))

desc = [item.text.strip() for item in soup.find_all('div', {'class': 'recipe__step-text'})]

