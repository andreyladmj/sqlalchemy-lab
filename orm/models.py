from datetime import datetime

from sqlalchemy import (Table, Column, Integer, Numeric, String, DateTime,
                        ForeignKey)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, backref
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

engine = create_engine('sqlite:///:memory:')

Session = sessionmaker(bind=engine)

session = Session()

Base = declarative_base()


class Cookie(Base):
    __tablename__ = 'cookies'

    cookie_id = Column(Integer(), primary_key=True)
    cookie_name = Column(String(50), index=True)
    cookie_recipe_url = Column(String(255))
    cookie_sku = Column(String(55))
    quantity = Column(Integer())
    unit_cost = Column(Numeric(12, 2))

    def __repr__(self):
        return "Cookie(cookie_name='{self.cookie_name}', " \
               "cookie_recipe_url='{self.cookie_recipe_url}', " \
               "cookie_sku='{self.cookie_sku}', " \
               "quantity={self.quantity}, " \
               "unit_cost={self.unit_cost})".format(self=self)


class User(Base):
    __tablename__ = 'users'

    user_id = Column(Integer(), primary_key=True)
    username = Column(String(15), nullable=False, unique=True)
    email_address = Column(String(255), nullable=False)
    phone = Column(String(20), nullable=False)
    password = Column(String(25), nullable=False)
    created_on = Column(DateTime(), default=datetime.now)
    updated_on = Column(DateTime(), default=datetime.now,
                        onupdate=datetime.now)

    def __repr__(self):
        return "User(username='{self.username}', " \
               "email_address='{self.email_address}', " \
               "phone='{self.phone}', " \
               "password='{self.password}')".format(self=self)

class Order(Base):
    __tablename__ = 'orders'
    order_id = Column(Integer(), primary_key=True)
    user_id = Column(Integer(), ForeignKey('users.user_id'))

    user =  relationship("User", backref=backref('orders', order_by=order_id))

    def __repr__(self):
        return "Order(user_id={self.user_id}, " \
               "shipped={self.shipped})".format(self=self)


class LineItems(Base):
    __tablename__ = 'line_items'
    line_item_id = Column(Integer(), primary_key=True)
    order_id = Column(Integer(), ForeignKey('orders.order_id'))
    cookie_id = Column(Integer(), ForeignKey('cookies.cookie_id'))
    quantity = Column(Integer())
    extended_cost = Column(Numeric(12, 2))
    order = relationship("Order", backref=backref('line_items',
                                                  order_by=line_item_id))
    cookie = relationship("Cookie", uselist=False, order_by=id)

    def __repr__(self):
        return "LineItems(order_id={self.order_id}, " \
               "cookie_id={self.cookie_id}, " \
               "quantity={self.quantity}, " \
               "extended_cost={self.extended_cost})".format(
            self=self)


Base.metadata.create_all(engine)




dcc = Cookie(cookie_name='dark chocolate chip',
             cookie_recipe_url='http://some.aweso.me/cookie/recipe_dark.html',
             cookie_sku='CC02',
             quantity=1,
             unit_cost=0.75)
mol = Cookie(cookie_name='molasses',
             cookie_recipe_url='http://some.aweso.me/cookie/recipe_molasses.html',
             cookie_sku='MOL01',
             quantity=1,
             unit_cost=0.80)
session.add(dcc)
session.add(mol)
session.flush()
print(dcc.cookie_id)
print(mol.cookie_id)



c1 = Cookie(cookie_name='peanut butter',
            cookie_recipe_url='http://some.aweso.me/cookie/peanut.html',
            cookie_sku='PB01',
            quantity=24,
            unit_cost=0.25)
c2 = Cookie(cookie_name='oatmeal raisin',
            cookie_recipe_url='http://some.okay.me/cookie/raisin.html',
            cookie_sku='EWW01',
            quantity=100,
            unit_cost=1.00)
session.bulk_save_objects([c1,c2])
session.commit()
print(c1.cookie_id)



for cookie in session.query(Cookie):
    print(cookie)



# Use the iterable version of the query over the all() method. It is more memory efficient than handling a full list of objects and we tend to operate on the data one record at a time anyway.
print(session.query(Cookie.cookie_name, Cookie.quantity).first())



for cookie in session.query(Cookie).order_by(Cookie.quantity): # Cookie.quantity.desc()
    print('{:3} - {}'.format(cookie.quantity, cookie.cookie_name))

query = session.query(Cookie).order_by(Cookie.quantity)[:2]
print([result.cookie_name for result in query])
query = session.query(Cookie).order_by(Cookie.quantity).limit(2)
print([result.cookie_name for result in query])


# SUM
from sqlalchemy import func
inv_count = session.query(func.sum(Cookie.quantity)).scalar()
print(inv_count)


# LIKE operator
rec_count = session.query(func.count(Cookie.cookie_name).label('inventory_count')).first()
print(rec_count.keys())
print(rec_count.inventory_count)

query = session.query(Cookie).filter(Cookie.cookie_name.like('%chocolate%'))
for record in query:
    print(record.cookie_name)


# CAST operator
from sqlalchemy import cast
query = session.query(Cookie.cookie_name,
                      cast((Cookie.quantity * Cookie.unit_cost),
                           Numeric(12,2)).label('inv_cost'))
for result in query:
    print('{} - {}'.format(result.cookie_name, result.inv_cost))



# UPDATE
query = session.query(Cookie)
cc_cookie = query.filter(Cookie.cookie_name == "chocolate chip").first()
cc_cookie.quantity = cc_cookie.quantity + 120
session.commit()
print(cc_cookie.quantity)
# OR
query = session.query(Cookie)
query = query.filter(Cookie.cookie_name == "chocolate chip")
query.update({Cookie.quantity: Cookie.quantity - 20})
cc_cookie = query.first()
print(cc_cookie.quantity)


# DELETE
query = session.query(Cookie)
query = query.filter(Cookie.cookie_name == "molasses")
query.delete()



# JOINS
query = session.query(Order.order_id, User.username, User.phone,
                      Cookie.cookie_name, LineItem.quantity,
                      LineItem.extended_cost)
query = query.join(User).join(LineItem).join(Cookie)
results = query.filter(User.username == 'cookiemon').all()
print(results)


query = session.query(User.username, func.count(Order.order_id))
query = query.outerjoin(Order).group_by(User.username)
for row in query:
    print(row)



# SELF referential table
class Employee(Base):
    __tablename__ = 'employees'

    id = Column(Integer(), primary_key=True)
    manager_id = Column(Integer(), ForeignKey('employees.id'))
    name = Column(String(255), nullable=False)

    manager = relationship("Employee", backref=backref('reports'),
                           remote_side=[id])