import asyncio
import datetime

from rx import Observable
from rx.concurrency import IOLoopScheduler
from rx.subjects import Subject
from tornado.ioloop import IOLoop
from tornado.websocket import websocket_connect


class Client:
    def __init__(self, host='localhost', port='8000'):
        self._url = 'ws://{}:{}/exchange'.format(host, port)
        self.conn = None
        self.opened = Subject()
        self.messages = Subject()

    def connect(self):
        def on_connect(conn):
            self.conn = conn
            self.opened.on_next(conn)
            self.opened.on_completed()
            self.opened.dispose()

        def on_message_callback(message):
            self.messages.on_next(message)

        future = websocket_connect(self._url, on_message_callback=on_message_callback)
        Observable.from_future(future).subscribe(on_connect)

    def write_message(self, message):
        self.conn.write_message(message)

if __name__ == '__main__':
    scheluder = IOLoopScheduler(IOLoop.current())

    def make_say_hello(client, client_id):
        def say_hello():
            print("{} client #{} is sending orders".format())