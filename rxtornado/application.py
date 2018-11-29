from rx.concurrency import IOLoopScheduler
from rx.subjects import Subject
from tornado.ioloop import IOLoop
from tornado.web import RequestHandler, Application
from tornado.websocket import WebSocketHandler


class MainHandler(RequestHandler):
    def get(self):
        self.write('Hello Stock Excange!\n')

class ExchangeHandler(WebSocketHandler):
    def open(self):
        Server().messages.on_next(['opened', self.request])

    def on_message(self, message):
        Server().messages.on_next(['message', message])

    def on_close(self):
        Server().messages.on_next(['closed', self.request])

class Server:
    class __Server:
        def __init__(self):
            scheduler = IOLoopScheduler(IOLoop.current())
            self.messages = Subject()
            only_messages = self.messages.filter(lambda msg: msg[0] == 'message').map(lambda msg: msg[1]).publish()
            only_messages.subscribe(lambda msg: print(msg))
            only_messages.connect()
            self._app = Application([
                (r'/exchange', ExchangeHandler),
                (r'/', MainHandler),
            ])

        def start(self):
            self._app.listen(8000)

    instance = None

    def __init__(self):
        if Server.instance is None:
            Server.instance = Server.__Server()

    def __getattr__(self, item):
        return getattr(self.instance, item)

if __name__ == '__main__':
    Server().messages.subscribe(lambda msg: print('Received: {}'.format(msg)))
    Server().messages.filter(lambda msg: msg == 'opened').subscribe(lambda msg: print('Connection has been opened'))
    Server().start()
    IOLoop.current().start()