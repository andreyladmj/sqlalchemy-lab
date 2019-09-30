from threading import Thread

class threads_object(Thread):
    def run(self):
        function_to_run()



class non_threads_object(object):
    def run(self):
        function_to_run()


def non_threaded(num_iter):
    funcs = []
    for i in range(int(num_iter)):
        funcs.append(non_threads_object())
    for i in funcs:
        i.run()


def threaded(num_iter):
    funcs = []
    for i in range(int(num_iter)):
        funcs.append(threads_object())
    for i in funcs:
        i.start()
    for i in funcs:
        i.join()

def function_to_run():
    s = 0
    for i in range(1000):
        s += i
        f = open("guru99.txt", "w+")
        f.write(str(s))
        f.close()

def show_results(func_name, results):
    print("%-23s %4.6f seconds" % (func_name, results))

if __name__ == '__main__':
    import sys
    from timeit import Timer

    repeat = 100
    number = 1
    num_threads = [1,2,4,8]

print("start tersting")
for i in num_threads:
    t = Timer("non_threaded(%s)" % i, "from __main__ import non_threaded")
    best_result = min(t.repeat(repeat=repeat, number=number))
    show_results("non threaded (%s iters)" % i, best_result)

    t = Timer("threaded(%s)" % i, "from __main__ import threaded")
    best_result = min(t.repeat(repeat=repeat, number=number))
    show_results("threaded (%s iters)" % i, best_result)
    print('')