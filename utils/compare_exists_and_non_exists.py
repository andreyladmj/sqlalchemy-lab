# cython: profile=True

import pandas as pd
import numpy as np


def cccompare(batch, df_db_rows):
    exists_ids = df_db_rows.index.unique()

    new_df_data = pd.DataFrame.from_records(batch).set_index('writer_id')

    df_db_rows['new_observation_count'] = new_df_data.observation_count
    df_db_rows['new_metrics_value'] = new_df_data.metrics_value
    df_db_rows['new_std_error'] = new_df_data.std_error
    df_db_rows['new_top_quantile'] = new_df_data.top_quantile

    df_db_rows.loc[
        df_db_rows.new_std_error.apply(lambda x: not isinstance(x, float)), 'new_std_error'] = np.nan
    df_db_rows.loc[
        df_db_rows.metrics_value.apply(lambda x: not isinstance(x, float)), 'metrics_value'] = np.nan
    df_db_rows = df_db_rows.fillna(0)

    eq1 = (df_db_rows.observation_count != df_db_rows.new_observation_count)
    eq2 = ~(np.isclose(df_db_rows.metrics_value, df_db_rows.new_metrics_value, 1.e-5))
    eq3 = ~(np.isclose(df_db_rows.std_error, df_db_rows.new_std_error, 1.e-5))
    eq4 = ~(np.isclose(df_db_rows.top_quantile, df_db_rows.new_top_quantile, 1.e-5))

    update_ids = df_db_rows[eq1 | eq2 | eq3 | eq4].index

    rows_for_insert = new_df_data[~new_df_data.index.isin(exists_ids)].reset_index().to_dict('rows')
    rows_for_insert += new_df_data[new_df_data.index.isin(update_ids)].reset_index().to_dict('rows')