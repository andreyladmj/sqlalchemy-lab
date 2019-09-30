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

    metric = CancelRateMetric()
    start_date = datetime.strptime("2018-01-01", "%Y-%m-%d")
    end_date = datetime.strptime("2018-01-05", "%Y-%m-%d")

    stime = time()
    print("\n\nUpdating", metric.__class__, "\n\n")
    metric.load_raw_data(date_start=start_date,end_date=end_date)
    l = 0

    for partition in metric.iterate_by_period(dtime):
        partition.calculate_metric_score(min_obs_count=1)
        try:
            c = partition.df_wr.loc[60817]
            if l != c.observation_count:
                print(c.metrics_value, c.observation_count, c.metrics_value_std, c.date_observation)
            l = c.observation_count
        except:pass

    # df_wr.loc[60817]

    # metric.calculate_metric_score(min_obs_count=50)
    # print("From:", metric.df_raw.date_observation.min(), "To:", metric.df_raw.date_observation.max())
    # db.update_metric(metric)
    # db.update_writer_metric(metric)

    # for partition in metric.iterate_by_period(dtime):
    #     print("From:", partition.df_raw.date_observation.min(), "To:", partition.df_raw.date_observation.max())
    #     if not partition.df_raw.empty:
    #         partition.calculate_metric_score(min_obs_count=50)
    #         db.update_metric(partition)
    #         db.update_writer_metric(partition)
    #
    # print("Updated!", time() - stime)
