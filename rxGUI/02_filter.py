import random
import sys

from PyQt5 import QtCore
from PyQt5.QtWidgets import QTableWidget, QWidget, QGridLayout, QApplication, QTableWidgetItem
from rx import Observable
from rx.concurrency import QtScheduler
from rx.subjects import Subject


class StockOverviewTable(QTableWidget):
    def __init__(self, *args, **kwargs):
        stock_prices_stream = kwargs.pop('stock_prices_stream')
        QTableWidget.__init__(self, *args, **kwargs)
        self.setRowCount(0)
        self.setColumnCount(4)
        self.setHorizontalHeaderLabels(['Symbol', 'Name', 'Buy Price', 'Sell Price'])
        self.setColumnWidth(0, 50)
        self.setColumnWidth(1, 200)
        self.setColumnWidth(2, 100)
        self.setColumnWidth(3, 100)
        self.horizontalHeader().setStretchLastSection(True)
        self.setSortingEnabled(True)
        stock_prices_stream.subscribe(self._create_ot_update_stock_row)

    def _create_ot_update_stock_row(self, stock_row):
        row = self._find_matching_row_index(stock_row)
        column_index = 0
        for column in stock_row:
            self.setItem(row, column_index, QTableWidgetItem(str(column)))
            column_index += 1

    def _find_matching_row_index(self, stock_row):
        matches = self.findItems(stock_row[0], QtCore.Qt.MatchExactly)
        if len(matches) == 0:
            self.setRowCount(self.rowCount() + 1)
            return self.rowCount() - 1
        return self.indexFromItem(matches[0]).row()


class HelloWorld(QWidget):
    def __init__(self, *args, **kwargs):
        stock_prices_stream = kwargs.pop('stock_prices_stream')
        QWidget.__init__(self, *args, **kwargs)
        self.events = Subject()
        self._setup_window()
        self._layout = QGridLayout(self)
        self._overview_table = StockOverviewTable(stock_prices_stream=stock_prices_stream)
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
    scheduler = QtScheduler(QtCore)
    stock_prices = Observable.interval(REFRESH_STOCK_INTERVAL, scheduler).map(random_stock).publish()
    hello_world = HelloWorld(stock_prices_stream=stock_prices)
    hello_world.show()
    stock_prices.connect()
    sys.exit(app.exec_())