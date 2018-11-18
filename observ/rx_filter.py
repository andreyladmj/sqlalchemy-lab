from rx import Observable

Observable.from_(['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon']) \
    .map(lambda s: len(s)) \
    .filter(lambda i: i >= 5) \
    .take(2)\
    .take_while(lambda x: x < 100)\
    .distinct()\
    .subscribe(lambda x: print(x))

Observable.from_(['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon']) \
    .distinct(lambda s: len(s)) \
    .subscribe(lambda x: print(x))

Observable.from_(['Alpha', 'NNNNN', 'Beta', 'Gamma', 'Delta', 'Epsilon']) \
    .map(lambda s: len(s)) \
    .distinct_until_changed() \
    .distinct_until_changed(lambda s: len(s)) \
    .subscribe(lambda x: print(x))

Observable.from_(['Alpha', 'NNNNN', 'Beta', 'Gamma', 'Delta', 'Epsilon']) \
    .filter(lambda s: len(s) > 4) \
    .count() \
    .subscribe(lambda ct: print(ct))


Observable.from_([1, 2, 445, 46, 2, 23, 5]) \
    .filter(lambda i: i > 4) \
    .sum() \
    .subscribe(lambda ct: print(ct))

Observable.from_([1, 2, 445, 46, 2, 23, 5]) \
    .filter(lambda i: i < 100) \
    .reduce(lambda total, i: total + i) \
    .subscribe(lambda ct: print(ct))

Observable.from_([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) \
    .filter(lambda i: i < 100) \
    .scan(lambda total, i: total + i) \
    .subscribe(lambda ct: print(ct))

Observable.from_([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) \
    .to_dict(lambda i: i * 100) \
    .subscribe(lambda ct: print(ct))

# distinct - unique
