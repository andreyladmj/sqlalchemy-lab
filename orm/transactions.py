from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

engine = create_engine('sqlite:///:memory:')

Session = sessionmaker(bind=engine)

session = Session()

from datetime import datetime

from sqlalchemy import (Table, Column, Integer, Numeric, String,
                        DateTime, ForeignKey, Boolean, CheckConstraint)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, backref


Base = declarative_base()


class Cookie(Base):
    __tablename__ = 'cookies'
    __table_args__ = (CheckConstraint('quantity >= 0', name='quantity_positive'),)

    cookie_id = Column(Integer, primary_key=True)
    cookie_name = Column(String(50), index=True)
    cookie_recipe_url = Column(String(255))
    cookie_sku = Column(String(55))
    quantity = Column(Integer())
    unit_cost = Column(Numeric(12, 2))

    def __init__(self, name, recipe_url=None, sku=None, quantity=0, unit_cost=0.00):
        self.cookie_name = name
        self.cookie_recipe_url = recipe_url
        self.cookie_sku = sku
        self.quantity = quantity
        self.unit_cost = unit_cost

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
    updated_on = Column(DateTime(), default=datetime.now, onupdate=datetime.now)

    def __init__(self, username, email_address, phone, password):
        self.username = username
        self.email_address = email_address
        self.phone = phone
        self.password = password

    def __repr__(self):
        return "User(username='{self.username}', " \
               "email_address='{self.email_address}', " \
               "phone='{self.phone}', " \
               "password='{self.password}')".format(self=self)


class Order(Base):
    __tablename__ = 'orders'
    order_id = Column(Integer(), primary_key=True)
    user_id = Column(Integer(), ForeignKey('users.user_id'))
    shipped = Column(Boolean(), default=False)

    user =  relationship("User", backref=backref('orders', order_by=order_id))

    def __repr__(self):
        return "Order(user_id={self.user_id}, " \
               "shipped={self.shipped})".format(self=self)


class LineItem(Base):
    __tablename__ = 'line_items'
    line_item_id = Column(Integer(), primary_key=True)
    order_id = Column(Integer(), ForeignKey('orders.order_id'))
    cookie_id = Column(Integer(), ForeignKey('cookies.cookie_id'))
    quantity = Column(Integer())
    extended_cost = Column(Numeric(12, 2))

    order = relationship("Order", backref=backref('line_items',
                                                  order_by=line_item_id))
    cookie = relationship("Cookie", uselist=False)

    def __repr__(self):
        return "LineItem(order_id={self.order_id}, " \
               "cookie_id={self.cookie_id}, " \
               "quantity={self.quantity}, " \
               "extended_cost={self.extended_cost})".format(
            self=self)

Base.metadata.create_all(engine)

cookiemon = User('cookiemon', 'mon@cookie.com', '111-111-1111', 'password')
cc = Cookie('chocolate chip', 'http://some.aweso.me/cookie/recipe.html', 'CC01', 12, 0.50)
dcc = Cookie('dark chocolate chip', 'http://some.aweso.me/cookie/recipe_dark.html', 'CC02', 1, 0.75)

session.add(cookiemon)
session.add(cc)
session.add(dcc)




o1 = Order()
o1.user = cookiemon
session.add(o1)

line1 = LineItem(order=o1, cookie=cc, quantity=9, extended_cost=4.50)


session.add(line1)
session.commit()
o2 = Order()
o2.user = cookiemon
session.add(o2)

line1 = LineItem(order=o2, cookie=cc, quantity=2, extended_cost=1.50)
line2 = LineItem(order=o2, cookie=dcc, quantity=9, extended_cost=6.75)


session.add(line1)
session.add(line2)
session.commit()



def ship_it(order_id):
    order = session.query(Order).get(order_id)
    for li in order.line_items:
        li.cookie.quantity = li.cookie.quantity - li.quantity
        session.add(li.cookie)
    order.shipped = True
    session.add(order)
    session.commit()
    print("shipped order ID: {}".format(order_id))


ship_it(3)
print(session.query(Cookie.cookie_name, Cookie.quantity).all())
session.rollback()
print(session.query(Cookie.cookie_name, Cookie.quantity).all())

from sqlalchemy.exc import IntegrityError
def ship_it(order_id):
    order = session.query(Order).get(order_id)
    for li in order.line_items:
        li.cookie.quantity = li.cookie.quantity - li.quantity
        session.add(li.cookie)
    order.shipped = True
    session.add(order)
    try:
        session.commit()
        print("shipped order ID: {}".format(order_id))
    except IntegrityError as error:
        print('ERROR: {!s}'.format(error.orig))
        session.rollback()
ship_it(3)