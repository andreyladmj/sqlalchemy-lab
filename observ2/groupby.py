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
    groups = Observable.from_(range(3)).group_by(key_selector)
    print(groups)
    groups.subscribe(subscribe_group_observable)
    input("\n")