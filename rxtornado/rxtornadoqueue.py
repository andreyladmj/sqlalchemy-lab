from rx.concurrency import IOLoopScheduler
from rx.subjects import Subject
from tornado.ioloop import IOLoop
from tornado.queues import PriorityQueue
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
