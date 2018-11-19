from urllib.request import urlopen

from rx import Observable


def read_request(link):
    f = urlopen(link)

    return Observable.from_(f) \
        .map(lambda b: b.decode("utf-8").strip())


read_request('https://goo.gl/rIaDyM') \
    .subscribe(lambda l: print(l))
