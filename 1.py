from sqlalchemy import create_engine
engine = create_engine('mysql+pymysql://cookiemonster:chocolatechip' \
                       '@mysql01.monster.internal/cookies', pool_recycle=3600)
connection = engine.connect()


from datetime import datetime
from sqlalchemy import DateTime

users = Table('users', metadata,
              Column('user_id', Integer(), primary_key=True),
              Column('username', String(15), nullable=False, unique=True), #1
Column('email_address', String(255), nullable=False),
Column('phone', String(20), nullable=False),
Column('password', String(25), nullable=False),
Column('created_on', DateTime(), default=datetime.now), #2
Column('updated_on', DateTime(), default=datetime.now, onupdate=datetime.now) #3
)

CheckConstraint('unit_cost >= 0.00', name='unit_cost_positive')