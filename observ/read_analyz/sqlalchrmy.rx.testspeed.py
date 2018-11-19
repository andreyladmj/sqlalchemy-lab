from time import time

from edusson_ds_main.db.connections import DBConnectionsFacade
from rx import Observable
from sqlalchemy import create_engine, text

# engine = create_engine()
# conn = engine.connect()
conn = DBConnectionsFacade.get_edusson_ds().connect()

sql = text('SELECT * FROM edusson_data_science.ds_orders_backup;')

def get_all_customers():
    return Observable.from_(conn.execute(sql))


def check_rx():
    start_time = time()

    get_all_customers().subscribe(lambda t: print(t), on_completed=lambda: print('Completed!', time() - start_time))


def check_normal():
    start_time = time()
    for row in conn.execute(sql):
        print(row)
    print('Completed!', time() - start_time)


if __name__ == '__main__':
    # check_rx() # Completed! 26.85364603996277
    check_normal() # Completed! 18.805429458618164
