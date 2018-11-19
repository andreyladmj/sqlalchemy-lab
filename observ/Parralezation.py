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

Observable.from_(['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon', 'Teta', 'Era', 'Nove']) \
    .flat_map(lambda s: Observable.just(s).subscribe_on(pool_scheduler).map(lambda s: intense_calculation(s))) \
    .subscribe(on_next=lambda s: print("Process 1: {0} {1}".format(current_thread().name, s)),
               on_error=lambda e: print(e),
               on_completed=lambda: print('Process 1 is finished!'))

# switch_map - try to use