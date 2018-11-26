from rx import Observable

if __name__ == '__main__':

    Observable.from_(range(2000)).buffer(Observable.interval(0.1)).subscribe(lambda buffer: print('#', len(buffer)))