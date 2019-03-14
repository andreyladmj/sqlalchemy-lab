from datetime import timedelta
import pandas as pd

id_to_actions_count = {0: 'order_placed', 1: 'messages_count', 2: 'edits_count', 3: 'writer_approved_count',
                       4: 'canceled_bids_count', 5: 'paid_order_count', 6: 'chat_count'}
id_to_last_actions_dt = {0: 'dt_order_placed', 1: 'dt_last_message', 2: 'dt_last_edit', 3: 'dt_last_writer_approved',
                         4: 'dt_last_bid_cancel', 5: 'dt_last_paid_order', 6: 'dt_last_chat'}

times_df = None #= pd.read_pickle('/home/andrei/Python/sqlalchemy-lab/p2paid/times_df.pkl')


def process_partial_df(partial_df):
    return partial_df.groupby('order_id').apply(get_order_features)


def get_order_features(actions_df):
    paid_date = actions_df[actions_df.action_id == 5].dt_order_placed
    web_times = times_df.copy()

    if not paid_date.empty:
        paid_date = paid_date.iloc[0]
        web_times = times_df[times_df < paid_date]

    return pd.concat([get_features(actions_df, dt_order_placed) for dt_order_placed in web_times])


def get_features(df, dt_order_placed=timedelta(minutes=1)):
    df = df[~df.action_id.isin([0])].copy()
    df_actions_count = df.groupby('action_id').dt_order_placed.apply(lambda x: (x <= dt_order_placed).sum())
    df_actions_count = df_actions_count.rename(id_to_actions_count)

    df_last_action_dt = df.groupby('action_id').dt_order_placed.apply(lambda x: x[x <= dt_order_placed].max())
    mask = df_last_action_dt.notna()

    if mask.sum():
        df_last_action_dt.loc[mask] = dt_order_placed - df_last_action_dt[mask]

    df_last_action_dt = df_last_action_dt.rename(id_to_last_actions_dt)

    df_features = pd.concat((df_actions_count, df_last_action_dt))
    df_features.index.name = ''
    df_features = df_features.to_frame(dt_order_placed).T
    df_features.index.name = 'dt_order_placed'
    # print('Time', time()-stime, len(df))
    return df_features
