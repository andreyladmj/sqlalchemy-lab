from rx import Observable

Observable.from_(['abc', 'def', 'ghi']).subscribe(lambda x: print(x))


Observable.interval(1).take_until(Observable.timer(30)).sample(3).subscribe(lambda x: print('X', x))