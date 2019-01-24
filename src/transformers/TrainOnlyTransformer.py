import pandas as pd

from src.transformers.ABCTransformer import ABCTransformer


class TrainOnlyTransformer(ABCTransformer):

    def fit_transform(self, x):
        return self.fit(x)

    @staticmethod
    def _one_hot(df, column):
        df = pd.concat([df, pd.get_dummies(df[column], prefix=column)], axis=1)
        df.drop([column], axis=1, inplace=True)
        return df

    def transform(self, x):
        pass

    def fit(self, df):
        df.sample(frac=1).reset_index(drop=True)

        df = self._one_hot(df, 'feature_1')
        df = self._one_hot(df, 'feature_2')
        df = self._one_hot(df, 'feature_3')
        df['year'] = pd.to_datetime(df['first_active_month']).dt.year
        df['month'] = pd.to_datetime(df['first_active_month']).dt.month
        df['year'].fillna(2017, inplace=True)
        df['month'].fillna(df['month'].mean(), inplace=True)
        df.drop(['first_active_month'], axis=1, inplace=True)
        df.drop(['card_id'], axis=1, inplace=True)

        y = []
        if 'target' in df.columns:
            y = df.pop('target').values
        x = df.values

        return x, y
