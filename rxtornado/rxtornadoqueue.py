from rx import Observable
from rx.concurrency import IOLoopScheduler
from rx.subjects import Subject
from tornado.ioloop import IOLoop
from tornado.queues import PriorityQueue, QueueEmpty
from tornado.web import RequestHandler, Application
from tornado.websocket import WebSocketHandler


class MainHandler(RequestHandler):
    def get(self):
        self.write('Hello Stock Exchange!\n')


class ExchangeHandler(WebSocketHandler):
    def open(self):
        Server().messages.on_next(['opened', self.request])

    def on_message(self, message):
        Server().message.on_next(['message', message])

    def on_close(self, message):
        Server().message.on_next(['closed', self.request])


class Server:
    class __Server:
        def __init__(self):
            scheduler = IOLoopScheduler(IOLoop.current())
            self._app = Application([
                (r'/exchange', ExchangeHandler),
                (r'/', MainHandler),
            ])
            self.orders = PriorityQueue()
            self.posted_orders = []
            self.fulfilled_orders = []
            self.messages = Subject()
            only_messages = self.messages.filter(lambda msg: msg[0] == 'message')\
                .map(lambda msg: msg[1].split(','))\
                .publish()

            def queue_order(msg):
                self.orders.put(Order.from_list(msg))

            only_messages.filter(lambda msg: msg[0] == 'order')\
                .map(lambda msg: msg[1:])\
                .subscribe(queue_order)

            def process_order(time):
                try:
                    order = self.orders.get_nowait()
                    print('processing order: {} [{}]'.format(order, order.timestamp))
                    matching = None
                    for posted in self.posted_orders:
                        if posts.matches(order):
                            matching = posted
                            break

                    if matching is None:
                        self.posted_orders.append(order)
                        print('Could not find match, posted order count is {}'.format(len(self.posted_orders)))
                    else:
                        self.posted_orders.remove(posted)
                        self.fulfilled_orders.append(posted)
                        self.fulfilled_orders.append(order)
                        print('order filfilled: {}'.format(order))
                        print('fulfilled by: {}'.format(posted))
                except QueueEmpty:
                    pass

            Observable.interval(100, scheduler=scheduler).subscribe(process_order)
            only_messages.connect()

    instance = None

    def __init__(self):
        if Server.instance is None:
            Server.instance = Server.__Server()

    def __getattr__(self, item):
        return getattr(self.instance, item)

if __name__ == '__main__':
    Server().messages.filter(lambda msg: msg == 'opened').subscribe(lambda msg: print('Connection has been opened'))
    Server().start()