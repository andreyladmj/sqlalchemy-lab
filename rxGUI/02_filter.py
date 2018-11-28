from random import random

import sys
from PyQt5.QtWidgets import QTableWidget, QWidget, QGridLayout, QApplication
from rx.subjects import Subject


class StockOverviewTable(QTableWidget):
    def __init__(self, *args, **kwargs):
        QTableWidget.__init__(self, *args, **kwargs)
        self.setRowCount(1)
        self.setColumnCount(4)
        self.setHorizontalHeaderLabels(['Symbol', 'Name', 'Buy Price', 'Sell Price'])
        self.setColumnWidth(0, 50)
        self.setColumnWidth(1, 200)
        self.setColumnWidth(2, 100)
        self.setColumnWidth(3, 100)
        self.horizontalHeader().setStretchLastSection(True)
        self.setSortingEnabled(True)

class HelloWorld(QWidget):
    def __init__(self, *args, **kwargs):
        QWidget.__init__(self, *args, **kwargs)
        self.events = Subject()
        self._setup_window()
        self._layout = QGridLayout(self)
        self._overview_table = StockOverviewTable()
        self._layout.addWidget(self._overview_table, 0, 0)

    def _setup_window(self):
        self.resize(640, 320)
        self.move(350, 200)
        self.setWindowTitle('Hello Wordls')

def random_stock(x):
    symbols_names = [
        ['ABC', 'abc manufacturing'],
        ['DEF', 'Desert Inc'],
        ['GHI', 'Chi Chi Inc'],
        ['A', 'A plus consulting'],
        ['GS', 'Great Security'],
        ['GO', 'Go Go Consulting'],
    ]
    stock = random.choice(symbols_names)
    return [
        stock[0],
        stock[1],
        round(random.uniform(21, 22), 2),
        round(random.uniform(20, 21), 2),
    ]

REFRESH_STOCK_INTERVAL = 2000

if __name__ == '__main__':
    app = QApplication(sys.argv)
    hello_world = HelloWorld()
    hello_world.show()
    sys.exit(app.exec_())