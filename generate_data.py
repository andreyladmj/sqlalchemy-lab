from random import choice, randint

from faker import Faker
from sqlalchemy import create_engine, select, func, insert

from model import admins, metadata, clients, items, clients_items
from sql_inspecter import SQLInspecter

fake = Faker()


def get_admins(n=20):
    admins = []

    for i in range(n):
        admins.append({
            'name': fake.name(),
            'age': randint(20, 50),
            'phone': fake.phone_number(),
            'gender': choice(['male', 'female']),
            'state': fake.state(),
            'registered': fake.date_time(),
        })

    return admins


def get_clients(n=20):
    clients = []

    for i in range(n):
        clients.append({
            'name': fake.name(),
            'age': randint(20, 50),
            'phone': fake.phone_number(),
            'gender': choice(['male', 'female']),
            'state': fake.state(),
            'job': fake.job(),
            'company': fake.company(),
            'address': fake.address(),
            'balance': randint(20, 5000),
            'registered': fake.date_time(),
        })

    return clients


def get_items(n=20):
    items = []

    for i in range(n):
        items.append({
            'name': ''.join(fake.words(nb=1, ext_word_list=None)),
            'cost': randint(10, 12000),
            'count': randint(10, 40),
        })

    return items


def generate():
    admins_list = get_admins(15)
    items_list = get_items(30)
    clients_list = get_clients(50)

    # engine = create_engine('mysql+pymysql://root@159.69.44.90/edusson_tmp_lab', pool_recycle=3600)
    engine = create_engine('sqlite:///:memory:')
    metadata.create_all(engine)
    connection = engine.connect()

    connection.execute(admins.insert(), admins_list)

    for i in clients_list:
        i['admin_id'] = connection.execute(select([admins.c.id]).order_by(func.random()).limit(1)).scalar()

    connection.execute(clients.insert(), clients_list)
    connection.execute(items.insert(), items_list)

    # SQLI = SQLInspecter(connection)
    # SQLI.listen_before_cursor_execute()
    # for i in range(20):
    #     ins = insert(clients_items).values(
    #         client_id=connection.execute(select([clients.c.id]).order_by(func.random()).limit(1)).scalar(),
    #         item_id=connection.execute(select([items.c.id]).order_by(func.random()).limit(1)).scalar(),
    #         count=randint(1, 8)
    #     )
    #     result = connection.execute(ins)
    #     print(result.inserted_primary_key)
    for i in range(20):
        ins = insert(clients_items).values(
            client_id=select([clients.c.id]).order_by(func.random()).limit(1),
            item_id=select([items.c.id]).order_by(func.random()).limit(1),
            count=randint(1, 8)
        )
        result = connection.execute(ins)
        # print(result.inserted_primary_key)
    return connection
    # print(SQLI.statements_info())


if __name__ == '__main__':
    generate()
