from rx import Observable

if __name__ == '__main__':
    Observable.from_(range(6)).window_with_count(2).flat_map(lambda x: x).subscribe(lambda x: print(x))