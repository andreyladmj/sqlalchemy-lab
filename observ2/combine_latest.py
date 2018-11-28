from rx import Observable
from rx.testing import TestScheduler

if __name__ == '__main__':
    test_scheduler = TestScheduler()
    Observable.combine_latest(
        Observable.interval(1, test_scheduler).map(lambda x: 'a: {}'.format(x)),
        Observable.interval(2, test_scheduler).map(lambda x: 'b: {}'.format(x)),
        lambda a, b: '{}: {}'.format(a, b)
    ).take_until(Observable.timer(5)).subscribe(lambda x: print(x))
    # test_scheduler.start()


    Observable.zip(
        Observable.interval(1, test_scheduler).map(lambda x: 'a: {}'.format(x)),
        Observable.interval(2, test_scheduler).map(lambda x: 'b: {}'.format(x)),
        lambda a, b: '{}: {}'.format(a, b)
    ).take_until(Observable.timer(5)).subscribe(lambda x: print(x))
    test_scheduler.start()
    input()