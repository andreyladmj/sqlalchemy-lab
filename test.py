between(cleft, cright)

# Find where the column is between cleft and cright

# concat(column_two)
#
# Concatenate column with column_two
#
# distinct()
#
# Find only unique values for the column
#
# in_([list])
#
# Find where the column is in the list
#
# is_(None)
#
# Find where the column is None (commonly used for Null checks with None)
#
# contains(string)
#
# Find where the column has string in it (case-sensitive)
#
# endswith(string)
#
# Find where the column ends with string (case-sensitive)
#
# like(string)
#
# Find where the column is like string (case-sensitive)
#
# startswith(string)
#
# Find where the column begins with string (case-sensitive)
#
# ilike(string)
#
# Find where the column is like string (this is not case-sensitive)

if __name__ == '__main__':
    from sqlalchemy.orm import sessionmaker
    from edusson_ds_main.db.connections import DBConnectionsFacade, DB_EDUSSON_DS
    from edusson_ds_main.db.models import DashDashboardBoard
    DB_EDUSSON_DS.set_static_connection(pool_recycle=500, pool_size=10, max_overflow=0, engine='mysql+pymysql',
                                        host='159.69.44.90', db='edusson_tmp_lab', user='root', passwd='')

    enable_sql_logging()
    session = sessionmaker(bind=DBConnectionsFacade.get_edusson_ds())()

    # subquery = session.query(DashDashboardBoard.board_id).filter(DashDashboardBoard.model_tag == 'writer_biding_coefficient')

    # q = session.query(DashUserBoardViewLog).outerjoin(DashUser.boards).order_by(DashUser.user_id)
    q = session.query(DashUserBoardViewLog).join(DashUserBoardViewLog.board).filter(DashDashboardBoard.model_tag == 'writer_biding_coefficient')
    # q = session.query(DashUserBoardViewLog).join(DashUserBoardViewLog.board).order_by(DashUser.user_id)

    print(DBRepository.get_count(q))

    # users = session.query(DashUser).outerjoin(DashUser.boards).order_by(DashUser.user_id).all()
    # for u in users:
    #     print(to_dict(u))
    #
    for s in stmts:
        print(s)
        print('')

    print('Queries count: ', len(stmts))

    '''
    
class BaseQuery(Query):
    def count_star(self):
        count_query = (self.statement.with_only_columns([func.count()])
                       .order_by(None))
        return self.session.execute(count_query).scalar()
        '''