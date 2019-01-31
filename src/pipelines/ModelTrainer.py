import math

from src.pipelines.ABCTrainer import ABCTrainer
from sklearn.model_selection import cross_val_score


class ModelTrainer(ABCTrainer):

    def __init__(self, transformer, estimator, data_provider):
        self._transformer = transformer
        self._estimator = estimator
        self._data_provider = data_provider
        self.model = None

    def train(self, df):
        x, y = self._transformer.fit_transform(df)

        self.model = self._estimator.fit(x, y)

        return self.model

    def predict(self, df):
        x_test, _ = self._transformer.transform(df)

        predictions = self.model.predict(x_test)

        return predictions

    def train_and_save(self, df):
        model = self.train(df)

        self._data_provider.save_model(model)

    def train_and_predict(self, train_set, test_set):
        self.train(train_set)

        predictions = self.predict(test_set)

        self._data_provider.save_submission(test_set, predictions)

    def cross_validate_and_predict(self, train_set, test_set, cv):
        x, y = self._transformer.fit_transform(train_set)

        mean_error = math.sqrt(-cross_val_score(self._estimator, x, y, scoring='neg_mean_squared_error', cv=cv, n_jobs=5).mean())

        predictions = self.predict(test_set)

        self._data_provider.save_submission(test_set, predictions)

        return mean_error

