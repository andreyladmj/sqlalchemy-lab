

import logging
import sys
import os
os.environ['DB_ENV'] = 'local'
from os import path, getcwd

sys.path.append(getcwd())
from datetime import datetime, timedelta
from time import time

logging.basicConfig(level=logging.INFO)

import numpy as np
import pandas as pd
from sqlalchemy import select, MetaData, event, alias, func
from writer_portrait.db import DB
from writer_portrait.metrics.cancel_rate import CancelRateMetric
from writer_portrait.metrics.customer_rating import CustomerRatingMetric
from writer_portrait.metrics.fine_rate import FineRateMetric
from writer_portrait.metrics.plag_rate import PlagRateMetric
from writer_portrait.metrics.reassign_rate import ReassignRateMetric


class DBNULL: pass


def add_own_encoders(conn, cursor, query, *args):
    cursor.connection.encoders[np.int64] = lambda value, encoders: int(value)
    cursor.connection.encoders[np.float64] = lambda value, encoders: float(value)
    cursor.connection.encoders[pd.Timestamp] = lambda value, encoders: encoders[str](str(value.to_pydatetime()))
    cursor.connection.encoders[pd.Timedelta] = lambda value, encoders: value.total_seconds()
    cursor.connection.encoders[DBNULL] = lambda value, encoders: "NULL"
    cursor.connection.encoders[np.nan] = lambda value, encoders: "NULL"


from edusson_ds_main.db.connections import DBConnectionsFacade

engine = DBConnectionsFacade.get_edusson_ds()
event.listen(engine, "before_cursor_execute", add_own_encoders)

meta = MetaData()
meta.reflect(bind=engine, only=['ds_metrics', 'ds_writer_metrics'])

DSMetrics = meta.tables['ds_metrics']
DSWriterMetrics = meta.tables['ds_writer_metrics']

logger = logging.getLogger('DB')



from compare_exists_and_non_exists import cccompare


def getdbrows(ids, metric_id, connection):
    subq = alias(select([func.max(DSWriterMetrics.c.id)])
                 .where(DSWriterMetrics.c.writer_id.in_(ids))
                 .where(DSWriterMetrics.c.metrics_id == metric_id)
                 .group_by(DSWriterMetrics.c.writer_id))

    q = DSWriterMetrics.select().where(DSWriterMetrics.c.id.in_(subq))
    return pd.read_sql(q, connection, index_col='writer_id')

def update(metric, dtime):
    for partition in metric.iterate_by_period(dtime):
        logger.info(str(len(partition.df_raw)) +": From:" + str(partition.df_raw.date_observation.min()) + "  To:" + str(partition.df_raw.date_observation.max()))
        partition.calculate_metric_score()
        data = partition.get_writers_metrics()

        if not len(data):
            continue

        n = np.ceil(len(data) / 5000) + 1  # approx 5k in batch

        metric_id = partition.get_id()

        with engine.connect() as connection:
            transaction = connection.begin()

            try:
                for i, batch in enumerate(np.array_split(data, n)):

                    if not len(batch):
                        continue

                    stime = time()
                    affected_rows = 0
                    new_df_data = pd.DataFrame.from_records(batch).set_index('writer_id')
                    ids = new_df_data.index.unique()

                    df_db_rows = getdbrows(ids, metric_id, connection)

                    rows_for_insert = cccompare(batch, df_db_rows)

                    if rows_for_insert:
                        query = DSWriterMetrics.insert().values(rows_for_insert)
                        affected_rows = connection.execute(query).rowcount

                    logger.info(f"Batch: {i}, Time: {time()-stime}, Affected rows: {affected_rows} from {len(batch)}")

                transaction.rollback()
            except Exception as e:
                transaction.rollback()
                logger.error("There are an error in update_writer_metric. Transaction rollback...")
                logger.error(e)
                continue


if __name__ == '__main__':
    dtime = timedelta(days=1)
    start_date = datetime.strptime("2019-01-01", "%Y-%m-%d")
    end_date = datetime.strptime("2019-01-10", "%Y-%m-%d")
    i = 0
    db = DB()

    metric = CancelRateMetric()
    metric.load_raw_data(start_date, end_date)
    metric.df_raw = metric.df_raw[metric.df_raw.date_observation < end_date]
    update(metric, dtime)
