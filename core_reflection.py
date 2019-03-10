# encoding: utf8
from sqlalchemy import select, MetaData
from sqlalchemy.ext.declarative import declarative_base
from werkzeug.security import generate_password_hash, check_password_hash
import enum
from sqlalchemy.orm import relationship
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import Session



db = SQLAlchemy()

from edusson_ds_main.db.connections import DBConnectionsFacade
from sqlalchemy.ext.automap import automap_base

# Base = declarative_base()
#Base = automap_base()
#metadata = Base.metadata
#Base.prepare(DBConnectionsFacade.get_edusson_ds(), reflect=True)

meta = MetaData()
meta.reflect(bind=DBConnectionsFacade.get_edusson_ds())

print(meta.tables)

DashUser = meta.tables['dash_user']

if __name__ == '__main__':

    # for i in range(100):
    #     session = Session(DBConnectionsFacade.get_edusson_ds())
    #     for user in session.query(DashUser).limit(10):
    #         print(user.username)
    #     session.close()

    connection = DBConnectionsFacade.get_edusson_ds().connect()
    columns = [DashUser.c.user_id, DashUser.c.username]
    query = select(columns)
    query = query.select_from(DashUser)
    result = connection.execute(query).fetchall()
    for row in result:
        print(row)

    connection.close()