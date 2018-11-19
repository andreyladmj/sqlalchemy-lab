import time

from rx import Observable

source = Observable.from_(['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon']).publish()

source.subscribe(lambda x: print('Source 1', x))
source.subscribe(lambda x: print('Source 2', x))

source.connect()


# source = Observable.interval(1000).share()
source = Observable.interval(1000).publish()
source.connect()

source.subscribe(lambda s: print('Source 1', s))
time.sleep(5)
source.subscribe(lambda s: print('Source 2', s))
