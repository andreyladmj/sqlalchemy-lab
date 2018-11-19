import re

from rx import Observable


def words_from_file(file_name):
    file = open(file_name)

    return Observable.from_(file)\
        .flat_map(lambda line: Observable.from_(line.split(' ')).to_dict(lambda w: len(w)))

file_name = '/home/andrei/Python/sqlalchemy-lab/observ/read_analyz/test.txt'

words_from_file(file_name).subscribe(lambda w: print(w))