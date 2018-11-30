from rx import Observable

def print_val(val):
    print('val', val)

if __name__ == '__main__':
    pass
    # Observable.from_([1,2,3]).map(lambda x: [x - 1, x, x + 1]).subscribe(print_val)
    # Observable.from_([{'a':1},{'b':2},{'c':3}]).map(lambda x: {**x}).subscribe(print_val)
    # Observable.from_([{'a':1},{'b':2},{'c':3}]).map(lambda x: Observable.just(x)).merge_all().subscribe(print_val)
