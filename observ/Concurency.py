import random
import time
from threading import current_thread

import multiprocessing
from rx import Observable
from rx.concurrency import ThreadPoolScheduler


def intense_calculation(value):
    time.sleep(random.randint(5, 20) * .1)
    return value

optimal_thread_count = multiprocessing.cpu_count() + 1
pool_scheduler = ThreadPoolScheduler(optimal_thread_count)

print("We are using {0} threads".format(optimal_thread_count))

Observable.from_(['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon']) \
    .map(lambda s: intense_calculation(s)) \
    .subscribe_on(pool_scheduler) \
    .subscribe(on_next=lambda s: print("Process 1: {0} {1}".format(current_thread().name, s)),
               on_error=lambda e: print(e),
               on_completed=lambda: print('Process 1 is finished!'))

Observable.range(1, 10) \
    .map(lambda s: intense_calculation(s)) \
    .subscribe_on(pool_scheduler) \
    .subscribe(on_next=lambda s: print("Process 2: {0} {1}".format(current_thread().name, s)),
               on_error=lambda e: print(e),
               on_completed=lambda: print('Process 2 is finished!'))

# input('POU\n')

Observable.interval(1000) \
    .map(lambda i: i * 100) \
    .observe_on(pool_scheduler) \
    .map(lambda s: intense_calculation(s)) \
    .subscribe(on_next=lambda i: print("Process 3: {0} {1}".format(current_thread().name, i)))