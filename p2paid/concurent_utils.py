from datetime import timedelta
import pandas as pd

times_df = None

id_to_events_count = {
    0: 'order_placed',
    1: 'paid_order_count',
    2: 'writer_approved_count',
    3: 'messages_count',
    4: 'chat_count',
    5: 'edits_count',
    6: 'canceled_bids_count',
}
id_to_last_events_dt = {
    0: 'dt_order_placed',
    1: 'dt_last_paid_order',
    2: 'dt_last_writer_approved',
    3: 'dt_last_message',
    4: 'dt_last_chat',
    5: 'dt_last_edit',
    6: 'dt_last_bid_cancel',
}


def process_partial_df(partial_df):
    return partial_df.groupby('order_id').apply(get_order_features)


def get_order_features(events_df):
    paid_date = events_df[events_df.event_id == 5].dt_order_placed
    web_times = times_df.copy()

    if not paid_date.empty:
        paid_date = paid_date.iloc[0]
        web_times = times_df[times_df < paid_date]

    return pd.concat([get_features(events_df, dt_order_placed) for dt_order_placed in web_times])


def get_features(df, dt_order_placed=timedelta(minutes=1)):
    df = df[~df.event_id.isin([0])].copy()
    df_events_count = df.groupby('event_id').dt_order_placed.apply(lambda x: (x <= dt_order_placed).sum())
    df_events_count = df_events_count.rename(id_to_events_count)

    df_last_event_dt = df.groupby('event_id').dt_order_placed.apply(lambda x: x[x <= dt_order_placed].max())
    mask = df_last_event_dt.notna()

    if mask.sum():
        df_last_event_dt.loc[mask] = dt_order_placed - df_last_event_dt[mask]

    df_last_event_dt = df_last_event_dt.rename(id_to_last_events_dt)

    df_features = pd.concat((df_events_count, df_last_event_dt))
    df_features.index.name = ''
    df_features = df_features.to_frame(dt_order_placed).T
    df_features.index.name = 'dt_order_placed'
    # print('Time', time()-stime, len(df))
    return df_features
