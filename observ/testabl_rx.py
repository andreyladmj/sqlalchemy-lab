from rx import Observable, Observer

letters = Observable.from_(['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon'])

napped = letters.map(lambda s: len(s))

filtered = napped.filter(lambda i: i >= 5)

# class Subscribe(Observer):
#     def on_next(self, value):
#         print('on_next', value)
#
#     def on_completed(self):
#         print('on_completed')
#
#     def on_error(self, error):
#         print('on_error', error)
#
# letters.subscribe(Subscribe())
letters.subscribe(on_next=lambda x: print(x), on_completed=lambda: print('Done'))
filtered.subscribe(lambda x: print(x))


Observable.from_(['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon'])\
    .map(lambda s: len(s))\
    .filter(lambda i: i >= 5)\
    .subscribe(lambda x: print(x))


Observable.range(1, 10).subscribe(lambda x: print(x))
Observable.just('Test').subscribe(lambda x: print(x))


def push_numbers(observer):
    observer.on_next(300)
    observer.on_next(500)
    observer.on_next(700)
    observer.on_completed()

Observable.create(push_numbers).subscribe(lambda x: print(x), on_completed=lambda: print('CCompleted!'))




disposable = Observable.interval(1000)\
    .map(lambda i: "{0} Missisipi".format(i))\
    .subscribe(lambda x: print(x))

sleep(5)
disposable.dispose()