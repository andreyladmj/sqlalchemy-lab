import logging
from datetime import datetime, timedelta
from time import time

logging.basicConfig(level=logging.INFO)

from writer_portrait.db import DB
from writer_portrait.metrics.cancel_rate import CancelRateMetric
from writer_portrait.metrics.customer_rating import CustomerRatingMetric
from writer_portrait.metrics.fine_rate import FineRateMetric
from writer_portrait.metrics.plag_rate import PlagRateMetric
from writer_portrait.metrics.reassign_rate import ReassignRateMetric
#

def update_metric(metric, start_date, dtime):
    stime = time()
    print("\n\nUpdating", metric.__class__, "\n\n")
    metric.load_raw_data(date_start=start_date)

    for partition in metric.iterate_by_period(dtime):
        print("From:", partition.df_raw.date_observation.min(), "To:", partition.df_raw.date_observation.max())
        if not partition.df_raw.empty:
            partition.calculate_metric_score(min_obs_count=50)
            db.update_metric(partition)
            db.update_writer_metric(partition)

    print("Updated!", time() - stime)


if __name__ == '__main__':
    dtime = timedelta(hours=1)
    i = 0
    db = DB()

    metric = CustomerRatingMetric()
    start_date = datetime.strptime("2018-01-01", "%Y-%m-%d")
    # end_date = datetime.strptime("2018-08-29", "%Y-%m-%d")

    stime = time()
    print("\n\nUpdating", metric.__class__, "\n\n")
    metric.load_raw_data(date_start=start_date)

    # metric.calculate_metric_score(min_obs_count=50)
    # print("From:", metric.df_raw.date_observation.min(), "To:", metric.df_raw.date_observation.max())
    # db.update_metric(metric)
    # db.update_writer_metric(metric)

    for partition in metric.iterate_by_period(dtime):
        print("From:", partition.df_raw.date_observation.min(), "To:", partition.df_raw.date_observation.max())
        if not partition.df_raw.empty:
            partition.calculate_metric_score(min_obs_count=50)
            db.update_metric(partition)
            db.update_writer_metric(partition)

    print("Updated!", time() - stime)
