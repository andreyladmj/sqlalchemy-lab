import sys
from PyQt5.QtWidgets import QWidget, QPushButton, QApplication
from rx.subjects import Subject


class HelloWorld(QWidget):
    def __init__(self, *args, **kwargs):
        QWidget.__init__(self, *args, **kwargs)
        self.resize(250, 150)
        self.move(350, 200)
        self.setWindowTitle('Hello world')

        self.button = QPushButton('Hello', self)
        self.button.clicked.connect(self.button_clicked)
        self._times_clicked = 0

        self.events = Subject()

    def button_clicked(self):
        self._times_clicked += 1
        self.events.on_next({
            'source': 'hello_world',
            'data': 'clicked',
            'count': self._times_clicked,
        })

if __name__ == '__main__':
    app = QApplication(sys.argv)
    hello_world = HelloWorld()
    hello_world.show()
    hello_world.events.subscribe(lambda x: print(x))
    sys.exit(app.exec_())