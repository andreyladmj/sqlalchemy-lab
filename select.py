from sqlalchemy import select, and_, func

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


columns = [admins.c.id, admins.c.name]
query = select(columns)
query = query.select_from(admins)
connection = generate()
result = connection.execute(query).fetchall()
for row in result:
    print(row)

    # , and_(Switch.host_id==Host.id, Switch.deleted == False)


SQLI = SQLInspecter(connection)
SQLI.listen_before_cursor_execute()
SQLI.reset_statement()

columns = [admins.c.id, admins.c.name, clients.c.id, clients.c.name]
query = select(columns)
query = query.select_from(admins.join(clients, and_(clients.c.admin_id == admins.c.id, admins.c.name.like('L%'))))
result = connection.execute(query).fetchall()
for row in result:
    print(row)

SQLI.reset_statement()
columns = [admins.c.id, admins.c.name, func.count(clients.c.id)]
query = select(columns)
query = query.select_from(admins.join(clients, and_(clients.c.admin_id == admins.c.id, admins.c.name.like('L%')))).group_by(admins.c.id)
result = connection.execute(query).fetchall()
for row in result:
    print(row)
print(SQLI.statements_info())

print(admins.c.id, type(admins.c.id))
print(dir(admins.c.id))
print(admins.c.id.foreign_keys, admins.c.id.primary_key)
print(clients.c.admin_id.foreign_keys, clients.c.admin_id.primary_key)



# 'all_', 'anon_label', 'any_', 'append_foreign_key', 'asc', 'autoincrement', 'base_columns', 'between', 'bind', 'bool_op', 'cast', 'collate', 'comment', 'comparator', 'compare', 'compile', 'concat', 'constraints', 'contains', 'copy', 'default', 'desc', 'description', 'dispatch', 'distinct', 'doc', 'endswith', 'expression', 'foreign_keys', 'get_children', 'ilike', 'in_', 'index', 'info', 'is_', 'is_clause_element', 'is_distinct_from', 'is_literal', 'is_selectable', 'isnot', 'isnot_distinct_from', 'key', 'label', 'like', 'match', 'name', 'notilike', 'notin_', 'notlike', 'nullable', 'nullsfirst', 'nullslast', 'onupdate', 'op', 'operate', 'params', 'primary_key', 'proxy_set', 'quote', 'references', 'reverse_operate', 'self_group', 'server_default', 'server_onupdate', 'shares_lineage', 'startswith', 'supports_execution', 'system', 'table', 'timetuple', 'type', 'unique', 'unique_params']


columns = [admins.c.id, admins.c.name, clients.c.id, clients.c.name]
query = select(columns)
query = query.select_from(admins.join(clients, and_(clients.c.admin_id == admins.c.id, admins.c.name.like('L%'))))
result = connection.execute(query).fetchall()


Query(query).columns(admins.c.id, [admins.c.name]).has(clients.c.id, [clients.c.name])