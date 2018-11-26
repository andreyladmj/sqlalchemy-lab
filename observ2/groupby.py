from rx import Observable


def key_selector(x):
    if x % 2 == 0:
        return 'even'
    return 'odd'

def subscribe_group_observable(group_observable):
    def print_count(count):
        print('Groupd Observable key{} ,contain items {}'.format(group_observable.key, count))
    group_observable.count().subscribe(print_count)

if __name__ == '__main__':
    # groups = Observable.from_(range(3)).group_by(key_selector)
    # print(groups)
    # groups.subscribe(subscribe_group_observable)
    # input("\n")
    Observable.from_([
        {'cat': 1, 'items': 2},
        {'cat': 2, 'items': 1},
        {'cat': 3, 'items': 4},
        {'cat': 1, 'items': 1},
        {'cat': 2, 'items': 4},
    # ]).map(lambda t: Observable.just(t).group_by(lambda d: d['cat']).map(lambda d: d.sum(lambda d: d['items'])).flat_map(lambda d: d))\
    ])\
        .group_by(lambda d: d['cat'])\
        .map(lambda d: d.sum(lambda d: d['items']))\
        .flat_map(lambda d: d)\
        .subscribe(lambda s: print(s))



Observable.from_(list_) \
    .group_by(lambda s: len(s)) \
    .flat_map(lambda grp: grp.count().map(lambda ct: (grp.key, ct))) \
    .to_dict(lambda t: t[0], lambda t: t[1]) \
    .subscribe(lambda s: print(s))