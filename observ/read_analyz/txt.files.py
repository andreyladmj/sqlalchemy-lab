from rx import Observable


def read_lines(file_name):
    file = open(file_name)

    return Observable.from_(file)\
        .map(lambda l: l.strip())\
        .filter(lambda l: l != '')

read_lines('/home/andrei/Python/sqlalchemy-lab/observ/read_analyz/nicks.txt').subscribe(lambda l: print(l))