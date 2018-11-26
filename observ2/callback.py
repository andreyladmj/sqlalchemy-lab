from rx import Observable

def print_value(val):
    print(val)

def say_hellop(name, callback):
    callback('Hello {}'.format(name))

if __name__ == '__main__':
    hello = Observable.from_callback(say_hellop)
    hello('Jack').subscribe(print_value)
    hello('Sam').subscribe(print_value)