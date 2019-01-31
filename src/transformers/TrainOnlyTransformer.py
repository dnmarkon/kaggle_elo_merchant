import datetime

import pandas as pd

from src.misc.helpers import one_hot
from src.transformers.ABCTransformer import ABCTransformer


class TrainOnlyTransformer(ABCTransformer):

    def __init__(self):
        self.month_mean = 0.0

    def fit_transform(self, x):
        self.fit(x)

        return self.transform(x)

    def transform(self, df):
        df = one_hot(df, 'feature_1')
        df = one_hot(df, 'feature_2')
        df = one_hot(df, 'feature_3')
        df['first_active_month'].replace('0.0', datetime.datetime(2017,1,1), inplace=True)
        df['year'] = pd.to_datetime(df['first_active_month']).dt.year
        df['month'] = pd.to_datetime(df['first_active_month']).dt.month
        df['year'].fillna(2017, inplace=True)
        df['month'].fillna(self.month_mean, inplace=True)
        df.drop(['first_active_month'], axis=1, inplace=True)
        df.drop(['card_id'], axis=1, inplace=True)
        y = []
        if 'target' in df.columns:
            y = df.pop('target').values
        x = df.values

        return x, y

    def fit(self, df):
        df.sample(frac=1).reset_index(drop=True)

        self.month_mean = pd.to_datetime(df['first_active_month']).dt.month.mean()


