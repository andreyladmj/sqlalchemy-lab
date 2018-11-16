import enum
from datetime import datetime

from sqlalchemy import (MetaData, Table, Column, Integer, Numeric, String,
                        DateTime, ForeignKey, create_engine)
from sqlalchemy.dialects.mysql import TINYINT, ENUM, INTEGER

metadata = MetaData()


admins = Table('admins', metadata,
               Column('id', INTEGER(unsigned=True), primary_key=True, autoincrement=True),
               Column('name', String(50), index=True),
               Column('age', Integer()),
               Column('state', String(50)),
               Column('phone', String(50)),
               Column('gender', String(20)),
               Column('registered', DateTime())
               )

clients = Table('clients', metadata,
                Column('id', INTEGER(unsigned=True), primary_key=True, autoincrement=True),
                Column('admin_id', ForeignKey('admins.id')),
                Column('name', String(50), index=True),
                Column('age', Integer()),
                Column('phone', String(50)),
                Column('state', String(50)),
                Column('job', String(50)),
                Column('company', String(50)),
                Column('address', String(100)),
                Column('gender', String(20)),
                Column('balance', Numeric(12, 2)),
                Column('registered', DateTime())
                )

items = Table('items', metadata,
                Column('id', INTEGER(unsigned=True), primary_key=True, autoincrement=True),
                Column('name', String(50), index=True),
                Column('cost', Numeric(12, 2)),
                Column('count', INTEGER(unsigned=True)),
                )

clients_items = Table('clients_items', metadata,
                Column('id', INTEGER(unsigned=True), primary_key=True, autoincrement=True),
                Column('client_id', ForeignKey('clients.id')),
                Column('item_id', ForeignKey('items.id')),
                Column('count', INTEGER(unsigned=True)),
                )


engine = create_engine('sqlite:///:memory:')
metadata.create_all(engine)