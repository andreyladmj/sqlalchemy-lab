import random
import time
from threading import current_thread

import multiprocessing
from urllib.request import urlopen

import requests
from rx import Observable
from rx.concurrency import ThreadPoolScheduler

session = requests.Session()
session.auth = ("test", "Q8UtsemJHZEJxOnE")
hostname = "https://place2paid-prediction.edusson-data-science.com"


def read_request(i):
    response = session.get(hostname + '/api/v1/orders/{}'.format(i), timeout=3)
    # response = session.get(hostname + '/api/v1/orders/1092872', timeout=3)

    return Observable.from_(response) \
        .map(lambda b: response.content)


optimal_thread_count = multiprocessing.cpu_count() + 1
pool_scheduler = ThreadPoolScheduler(optimal_thread_count)

print("We are using {0} threads".format(optimal_thread_count))

Observable.range(1, 20) \
    .flat_map(lambda s: Observable.just(s).subscribe_on(pool_scheduler).map(lambda s: read_request(s))) \
    .subscribe(on_next=lambda s: print("Process 1: {0} {1}".format(current_thread().name, s)),
               on_error=lambda e: print(e),
               on_completed=lambda: print('Process 1 is finished!'))

Observable.range(1, 20) \
    .flat_map(lambda s: read_request(s).subscribe_on(pool_scheduler)) \
    .subscribe(on_next=lambda s: print("Process 1: {0} {1}".format(current_thread().name, s)),
               on_error=lambda e: print(e),
               on_completed=lambda: print('Process 1 is finished!'))

Observable.range(1, 50) \
    .flat_map(lambda s: Observable.just(s).subscribe_on(pool_scheduler).flat_map(lambda s: read_request(s))) \
    .subscribe(on_next=lambda s: print("Process 1: {0} {1}".format(current_thread().name, s)),
               on_error=lambda e: print(e),
               on_completed=lambda: print('Process 1 is finished!'))
# switch_map - try to use
