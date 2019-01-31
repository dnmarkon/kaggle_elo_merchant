from src.estimators.ABCEstimator import ABCEstimator
import lightgbm as lgb


class LightGBMEstimator(ABCEstimator):

    def __init__(self, params, num_boost_round):
        self.estimator = None
        self.params = params
        self.num_boost_round = num_boost_round

    def fit(self, x, y):
        train = lgb.Dataset(x, label=y, free_raw_data=False)
        self.estimator = lgb.train(params=self.params,
                                   train_set=train,
                                   num_boost_round=self.num_boost_round)
        return self.estimator

    def transform(self, x, y):
        pass

    def predict(self, x):
        self.estimator.predict(x, num_iteration=self.estimator.best_iteration)
