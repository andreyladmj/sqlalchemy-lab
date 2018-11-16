from datetime import datetime

from sqlalchemy import (MetaData, Table, Column, Integer, Numeric, String,
                        DateTime, ForeignKey, create_engine)
metadata = MetaData()

cookies = Table('cookies', metadata,
                Column('cookie_id', Integer(), primary_key=True),
                Column('cookie_name', String(50), index=True),
                Column('cookie_recipe_url', String(255)),
                Column('cookie_sku', String(55)),
                Column('quantity', Integer()),
                Column('unit_cost', Numeric(12, 2))
                )

users = Table('users', metadata,
              Column('user_id', Integer(), primary_key=True),
              Column('customer_number', Integer(), autoincrement=True),
              Column('username', String(15), nullable=False, unique=True),
              Column('email_address', String(255), nullable=False),
              Column('phone', String(20), nullable=False),
              Column('password', String(25), nullable=False),
              Column('created_on', DateTime(), default=datetime.now),
              Column('updated_on', DateTime(), default=datetime.now, onupdate=datetime.now)
              )

orders = Table('orders', metadata,
               Column('order_id', Integer(), primary_key=True),
               Column('user_id', ForeignKey('users.user_id'))
               )

line_items = Table('line_items', metadata,
                   Column('line_items_id', Integer(), primary_key=True),
                   Column('order_id', ForeignKey('orders.order_id')),
                   Column('cookie_id', ForeignKey('cookies.cookie_id')),
                   Column('quantity', Integer()),
                   Column('extended_cost', Numeric(12, 2))
                   )

engine = create_engine('sqlite:///:memory:')
metadata.create_all(engine)



ins = cookies.insert().values(
    cookie_name="chocolate chip",
    cookie_recipe_url="http://some.aweso.me/cookie/recipe.html",
    cookie_sku="CC01",
    quantity="12",
    unit_cost="0.50"
)
print(str(ins))
ins.compile().params
result = connection.execute(ins)
result.inserted_primary_key


from sqlalchemy import insert
ins = insert(cookies).values(
    cookie_name="chocolate chip",
    cookie_recipe_url="http://some.aweso.me/cookie/recipe.html",
    cookie_sku="CC01",
    quantity="12",
    unit_cost="0.50"
)

ins = cookies.insert()
inventory_list = [
    {
        'cookie_name': 'peanut butter',
        'cookie_recipe_url': 'http://some.aweso.me/cookie/peanut.html',
        'cookie_sku': 'PB01',
        'quantity': '24',
        'unit_cost': '0.25'
    },
    {
        'cookie_name': 'oatmeal raisin',
        'cookie_recipe_url': 'http://some.okay.me/cookie/raisin.html',
        'cookie_sku': 'EWW01',
        'quantity': '100',
        'unit_cost': '1.00'
    }
]
result = connection.execute(ins, inventory_list)

# Querying Data
from sqlalchemy.sql import select

s = select([cookies])
rp = connection.execute(s)
results = rp.fetchall()

s = cookies.select()
rp = connection.execute(s)
results = rp.fetchall()

first_row = results[0]
first_row[1]
first_row.cookie_name
first_row[cookies.c.cookie_name]

rp = connection.execute(s)
for record in rp:
    print(record.cookie_name)
rowcount()

first()
#Returns the first record if there is one and closes the connection.

fetchone()
#Returns one row, and leaves the cursor open for you to make additional fetch calls.

scalar()
#Returns a single value if a query results in a single record with one column.


# When writing production code, you should follow these guidelines:
#
# Use the first method for getting a single record over both the fetchone and scalar methods, because it is clearer to our fellow coders.
#
# Use the iterable version of the ResultProxy over the fetchall and fetchone methods. It is more memory efficient and we tend to operate on the data one record at a time.
#
# Avoid the fetchone method, as it leaves connections open if you are not careful.
#
# Use the scalar method sparingly, as it raises errors if a query ever returns more than one row with one column, which often gets missed during testing.

s = select([cookies.c.cookie_name, cookies.c.quantity])
rp = connection.execute(s)
print(rp.keys())
result = rp.first()




from sqlalchemy import desc
s = select([cookies.c.cookie_name, cookies.c.quantity])
s = s.order_by(desc(cookies.c.quantity))
s = s.order_by(cookies.c.quantity.desc())


s = select([func.sum(cookies.c.quantity)])
rp = connection.execute(s)
print(rp.scalar())


from sqlalchemy import and_, or_, not_
s = select([cookies]).where(
    and_(
        cookies.c.quantity > 23,
        cookies.c.unit_cost < 0.40
    )
)

from sqlalchemy import and_, or_, not_
s = select([cookies]).where(
    or_(
        cookies.c.quantity.between(10, 50),
        cookies.c.cookie_name.contains('chip')
    )
)



############################### UPDATE #############################
from sqlalchemy import update
u = update(cookies).where(cookies.c.cookie_name == "chocolate chip")
u = u.values(quantity=(cookies.c.quantity + 120))
result = connection.execute(u)
print(result.rowcount)
s = select([cookies]).where(cookies.c.cookie_name == "chocolate chip")
result = connection.execute(s).first()
for key in result.keys():
    print('{:>20}: {}'.format(key, result[key]))



############################### DELETE #############################
from sqlalchemy import delete
u = delete(cookies).where(cookies.c.cookie_name == "dark chocolate chip")
result = connection.execute(u)
print(result.rowcount)

s = select([cookies]).where(cookies.c.cookie_name == "dark chocolate chip")
result = connection.execute(s).fetchall()
print(len(result))

if __name__ == '__main__':
    pass