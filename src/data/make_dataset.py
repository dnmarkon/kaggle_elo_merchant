import datetime

import pandas as pd

import os

from src.misc.helpers import one_hot


def create_features(transactions):
    transactions['authorized_flag'] = transactions['authorized_flag'].map({'Y': 1, 'N': 0}).astype(int)
    transactions['category_1'] = transactions['category_1'].map({'Y': 1, 'N': 0}).astype(int)
    transactions['purchase_date'] = pd.to_datetime(transactions['purchase_date'])
    transactions['purchase_month'] = transactions['purchase_date'].dt.month
    transactions = one_hot(transactions, 'category_2')
    transactions = one_hot(transactions, 'category_3')
    transactions['month_diff'] = (datetime.datetime(2019, 1, 1) - transactions['purchase_date']).dt.days // 30

    return transactions


def aggregate_per_card(transactions):
    agg_func = {
        'authorized_flag': ['mean'],
        'category_1': ['mean'],
        'category_2_1.0': ['mean'],
        'category_2_2.0': ['mean'],
        'category_2_3.0': ['mean'],
        'category_2_4.0': ['mean'],
        'category_2_5.0': ['mean'],
        'category_3_A': ['mean'],
        'category_3_B': ['mean'],
        'category_3_C': ['mean'],
        'merchant_id': ['nunique'],
        'merchant_category_id': ['nunique'],
        'state_id': ['nunique'],
        'city_id': ['nunique'],
        'subsector_id': ['nunique'],
        'purchase_amount': ['sum', 'mean', 'max', 'min', 'std'],
        'installments': ['sum', 'mean', 'max', 'min', 'std'],
        'purchase_month': ['mean', 'max', 'min', 'std'],
        'month_lag': ['mean', 'max', 'min', 'std'],
        'month_diff': ['mean', 'max', 'min', 'std']
    }

    agg_history = transactions.groupby(['card_id']).agg(agg_func)
    agg_history.columns = ['_'.join(col).strip() for col in agg_history.columns]
    agg_history.reset_index(inplace=True)

    df = (transactions.groupby('card_id').size().reset_index(name='trans_count'))

    agg_history = pd.merge(df, agg_history, on='card_id', how='left')

    return agg_history


def merge_with_dataset(df, aggregs, join_type):
    for aggregation in aggregs:
        df = pd.merge(df, aggregation, how=join_type, on='card_id')

    return df


def expand_dataset(input_path, aggrs, join_type, output_path):
    df = pd.read_csv(input_path)
    df['first_active_month'] = pd.to_datetime(df['first_active_month'])
    df_aggr = merge_with_dataset(df, aggrs, join_type)
    df_aggr.fillna(0.0, inplace=True)
    df_aggr.to_csv(output_path, index=False)


if __name__ == '__main__':
    current_dir = os.getcwd()
    raw_path = os.path.join(current_dir, os.pardir, os.pardir, 'data', 'raw')
    interim_path = os.path.join(current_dir, os.pardir, os.pardir, 'data', 'interim')
    train_path = os.path.join(raw_path, 'train.csv')
    test_path = os.path.join(raw_path, 'test.csv')
    historical_path = os.path.join(raw_path, 'historical_transactions.csv')
    new_path = os.path.join(raw_path, 'new_merchant_transactions.csv')
    train_output_path = os.path.join(interim_path, 'train_with_new_aggregations.csv')
    test_output_path = os.path.join(interim_path, 'test_with_new_aggregations.csv')

    hist_trans = pd.read_csv(historical_path)
    new_trans = pd.read_csv(new_path)

    hist_trans = create_features(hist_trans)
    new_trans = create_features(new_trans)
    aggregated_hist_transactions = aggregate_per_card(hist_trans)
    aggregated_new_transactions = aggregate_per_card(new_trans)

    aggregations = [aggregated_hist_transactions, aggregated_new_transactions]

    expand_dataset(train_path, aggregations, 'inner', train_output_path)
    expand_dataset(test_path, aggregations, 'left', test_output_path)
