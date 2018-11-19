from rx import Observable

# https://www.safaribooksonline.com/videos/reactive-python-for/9781491979006/9781491979006-video294990

obs1 = Observable.from_([1, 2, 445, 46, 2, 23, 5])
obs2 = Observable.from_([2, 3, 88, 14, 7, 1, 41])

Observable.merge(obs1, obs2).subscribe(lambda x: print(x))

obs1 = Observable.interval(1000).map(lambda i: "Source 1: {0}".format(i))
obs2 = Observable.interval(500).map(lambda i: "Source 2: {0}".format(i))
obs3 = Observable.interval(300).map(lambda i: "Source 3: {0}".format(i))

Observable.merge(obs1, obs2, obs3) \
    .subscribe(lambda x: print(x))

Observable.from_([obs1, obs2, obs3]) \
    .merge_all() \
    .subscribe(lambda x: print(x))

items = ['"12/123/345/123/3/6', "8/3/1/6/9/05/", "4/3/6/8/9/4/3/67"]

Observable.from_(items) \
    .map(lambda s: Observable.from_(s.split('/'))) \
    .merge_all() \
    .subscribe(lambda i: print(i))

Observable.from_(items) \
    .flat_map(lambda s: Observable.from_(s.split('/'))) \
    .subscribe(lambda i: print(i))

Observable.concat(obs1, obs2).subscribe(lambda x: print(x))


letters = Observable.from_(['A', 'B', 'C', 'D', 'E', 'F'])
numbers = Observable.range(1,5)

Observable.zip(letters, numbers, lambda l,n: "{0}-({1})".format(l,n))\
    .subscribe(lambda x: print(x))

letters.zip(numbers, lambda l,n: "{0}-({1})".format(l,n))\
    .subscribe(lambda x: print(x))


letters = Observable.from_(['Alpha', 'Betta', 'Gamma', 'Delta', 'Epsilon'])
intervals = Observable.interval(1000)

Observable.zip(letters, intervals, lambda l,i: l)\
    .subscribe(lambda s: print(s), on_completed=lambda: print('Completed!'))


list_ = ['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon']


Observable.from_(list_)\
    .group_by(lambda s: len(s))\
    .flat_map(lambda grp: grp.to_list())\
    .subscribe(lambda s: print(s))

Observable.from_(list_)\
    .group_by(lambda s: len(s))\
    .flat_map(lambda grp: grp.count().map(lambda ct: (grp.key, ct)))\
    .to_dict(lambda t: t[0], lambda t: t[1])\
    .subscribe(lambda s: print(s))
