import pandas as pd

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 200)

df = pd.read_csv("D:\datasets\ml-25m\movies.csv")