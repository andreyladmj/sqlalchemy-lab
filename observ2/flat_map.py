from rx import Observable


def read_last_line(filename):
    with open(filename) as file:
        lines = file.readlines()
        return Observable.just(lines[-1])

if __name__ == '__main__':
    read_last_line('100 Sales Records.csv').flat_map(lambda line: read_last_line('100 Sales Records 2.csv')).subscribe(lambda x: print(x))