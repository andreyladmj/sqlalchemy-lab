from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

engine = create_engine('sqlite:///:memory:')

Session = sessionmaker(bind=engine)

session = Session()

from datetime import datetime

from sqlalchemy import (Table, Column, Integer, Numeric, String, DateTime,
                        ForeignKey, Boolean)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, backref


Base = declarative_base()


class Cookie(Base):
    __tablename__ = 'cookies'

    cookie_id = Column(Integer, primary_key=True)
    cookie_name = Column(String(50), index=True)
    cookie_recipe_url = Column(String(255))
    cookie_sku = Column(String(55))
    quantity = Column(Integer())
    unit_cost = Column(Numeric(12, 2))

    def __init__(self, name, recipe_url=None, sku=None, quantity=0,
                 unit_cost=0.00):
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
        return "LineItems(order_id={self.order_id}, " \
               "cookie_id={self.cookie_id}, " \
               "quantity={self.quantity}, " \
               "extended_cost={self.extended_cost})".format(
            self=self)

Base.metadata.create_all(engine)



cc_cookie = Cookie('chocolate chip',
                   'http://some.aweso.me/cookie/recipe.html',
                   'CC01', 12, 0.50)

from sqlalchemy import inspect
insp = inspect(cc_cookie)

for state in ['transient', 'pending', 'persistent', 'detached']:
    print('{:>10}: {}'.format(state, getattr(insp, state)))

for attr, attr_state in insp.attrs.items():
    if attr_state.history.has_changes():
        print('{}: {}'.format(attr, attr_state.value))
        print('History: {}\n'.format(attr_state.history))



from sqlalchemy.orm.exc import MultipleResultsFound, NoResultFound
try:
    results = session.query(Cookie).one()
except MultipleResultsFound as error:
    print('We found too many cookies... is that even possible?')
except NoResultFound as error:
    print('We found no cookies... is that even possible?')






session.add(cc_cookie)
cc_cookie.cookie_name = 'Change chocolate chip'
session.commit()
# session.flush()
# session.rollback()


cookiemon = User('cookiemon', 'mon@cookie.com', '111-111-1111', 'password')
session.add(cookiemon)
o1 = Order()
o1.user = cookiemon
session.add(o1)

cc = session.query(Cookie).filter(Cookie.cookie_name ==
                                  "Change chocolate chip").one()
line1 = LineItem(order=o1, cookie=cc, quantity=2, extended_cost=1.00)

session.add(line1)
session.commit()


order = session.query(Order).first()
session.expunge(order)
order.line_items