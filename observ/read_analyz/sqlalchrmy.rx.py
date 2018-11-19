from rx import Observable
from sqlalchemy import create_engine, text

engine = create_engine()
conn = engine.connect()


def get_all_customers():
    stmt = text('SELECT * FROM CUSTOMER')
    return Observable.from_(conn.execute(stmt))


get_all_customers().subscribe(lambda t: print(t))


def customer_for_id(customer_id):
    stmt = text("SELECT * FROM CUSTOMER WHERE customer_id = :id")
    return Observable.from_(conn.execute(stmt, id=customer_id))


Observable.from_([1, 3, 5]) \
    .flat_map(lambda i: customer_for_id(i)) \
    .subscribe(lambda t: print(t))
