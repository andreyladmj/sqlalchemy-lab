from sqlalchemy import event


class SQLInspecter:
    def __init__(self, connection):
        self.statements = []
        self.connection = connection

    def listen_before_cursor_execute(self):
        event.listen(self.connection, "before_cursor_execute", self.catch_queries)

    def catch_queries(self, conn, cursor, statement, *args, **kwargs):
        self.statements.append(statement)

    def reset_statement(self):
        self.statements = []

    def statements_info(self):
        print('------------SQL----------------')
        for stmts in self.statements:
            print(stmts)
        print('------------SQL----------------')