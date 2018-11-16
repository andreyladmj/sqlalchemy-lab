from sqlalchemy import select

from generate_data import generate
from model import admins, clients, items, clients_items
from sql_inspecter import SQLInspecter

columns = [admins.c.id, admins.c.name, clients.c.id, clients.c.name, items.c.name, items.c.cost, items.c.count]
query = select(columns)

if __name__ == '__main__':
    connection = generate()

    SQLI = SQLInspecter(connection)
    SQLI.listen_before_cursor_execute()

    query = query.select_from(admins.join(clients).join(clients_items).join(items))

    result = connection.execute(query).fetchall()

    for row in result:
        print(row)
    print(SQLI.statements_info())