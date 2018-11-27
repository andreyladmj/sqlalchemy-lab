import csv

from rx import Observable

filename = '100 Sales Records.csv'

def read_csv_file(filename):
    with open(filename) as file:
        lines = file.readlines()
        return lines[0], csv.reader(lines[1:])

# Region	Country	Item Type	Sales Channel	Order Priority	Order Date	Order ID	Ship Date	Units Sold	Unit Price	Unit Cost	Total Revenue	Total Cost	Total Profit

def csv_row_to_dict(row):
    return {
        'Region': row[0],
        'Region': row[0],
    }

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

    def jj(row):
        print('jjrow', row.to_list())
        return row.to_list()

    # Observable.from_(reader).to_dict(to_d).subscribe(lambda r: print(r))

    # Observable.zip(Observable.just(header), reader, lambda l,n: "{0}-({1})".format(l,n)) \
    #     .subscribe(lambda x: print(x))

    # Observable.from_(reader).to_dict(lambda row: {k:v for k,v in zip(header, row)}).subscribe(lambda r: print(r))
    source = Observable.from_iterable(reader)\
        .map(lambda row: Observable.zip(Observable.from_(header), row, lambda k,v: {k:v}).merge_all())\
        .map(jj) \
        .subscribe(lambda r: print(r, end="\n\n\n"))

    source.publish()

    # .flat_map(ff) \
    # .flat_map(ff) \
        # .to_dict(lambda row: 1).subscribe(lambda r: print(r))
    #.map(lambda row: Observable.from_(row).subscribe(lambda t: print('t', t)))\
    # .flat_map(lambda t: t.merge_all()) \
