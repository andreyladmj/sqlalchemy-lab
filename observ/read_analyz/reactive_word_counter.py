import re

from rx import Observable


def words_from_file(file_name):
    file = open(file_name)

    return Observable.from_(file) \
        .flat_map(lambda line: Observable.from_(line.split(' '))) \
        .map(lambda w: re.sub(r'[^\w]', '', w)) \
        .filter(lambda w: w != '') \
        .map(lambda w: w.lower())


def word_counter(file_name):
    return words_from_file(file_name) \
        .group_by(lambda w: w) \
        .flat_map(lambda grp: grp.count().map(lambda ct: (grp.key, ct)))


def word_counter_as_dict(file_name):
    return word_counter(file_name) \
        .to_dict(lambda x: x[0], lambda x: x[1])


file_name = '/home/andrei/Python/sqlalchemy-lab/observ/read_analyz/test.txt'

# word_counter(file_name).subscribe(lambda w: print(w))
Observable.interval(3000) \
    .flat_map(lambda i: word_counter_as_dict(file_name)) \
    .distinct_until_changed() \
    .subscribe(lambda w: print(w))
