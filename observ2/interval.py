from rx import Observable
from rx.testing import TestScheduler

if __name__ == '__main__':
    test_scheduler = TestScheduler()

    Observable.interval(10, test_scheduler).take_until(Observable.timer(30)).subscribe(lambda s: print(s))
    test_scheduler.start()