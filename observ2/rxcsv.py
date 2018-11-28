import csv

from rx import Observable

filename = '100 Sales Records.csv'

def read_csv_file(filename):
    with open(filename) as file:
        lines = file.readlines()
        return lines[0], csv.reader(lines[1:])

if __name__ == '__main_1_':
    Observable.from_([{'1':2}, {'2':7}, {'5':3}]).merge_all().subscribe(lambda g: print(g))

if __name__ == '__main__':
    header, reader = read_csv_file(filename)
    header = header.split(',')
    # f = list(reader)[0]
    #
    # for i, t in zip(f, header):
    #     print(i, t)

    # print({k:v for k,v in zip(header, f)})

    def to_d(row):
        print(row)
        print(len(header), len(row))
        d = {}
        for k,v in zip(header, row):
            # print(k, ' == ', v)
            d[k] = v

        print(d)
        return d

    def ff(row):
        print('row', row)
        return row.to_list()
        # Observable.from_(row).merge().subscribe(lambda g: print('g', g))
        Observable.merge(row[:]).subscribe(lambda g: print('g', g))
        # return row.flat_map(lambda i: i.flat_map(lambda y:y.flat_map(lambda o:o)))
        return Observable.merge(row[:])
        return row.merge_all()

    def subdictes_to_dict(row):
        d = {}

        for x in row:
            d.update(x)

        return d

    source = Observable.from_iterable(reader)\
        .map(lambda row: Observable.zip(Observable.from_(header), row, lambda k,v: {k:v}))\
        .map(lambda x: x.to_list().map(subdictes_to_dict))#.subscribe(lambda r: print(r, end="\n\n\n"))

    print(source.flat_map(lambda x: x).subscribe(lambda x: print(x)))
    # Observable.merge(source[:]).merge_all()
