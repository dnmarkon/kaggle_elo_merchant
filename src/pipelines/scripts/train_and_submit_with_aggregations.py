import pandas as pd
import os

from src.data.FileDataProvider import FileDataProvider
from src.estimators.LightGBMEstimator import LightGBMEstimator
from src.pipelines.ModelTrainer import ModelTrainer
from src.transformers.TrainOnlyTransformer import TrainOnlyTransformer

if __name__ == '__main__':
    current_dir = os.getcwd()
    root_path = os.path.join(current_dir, os.pardir, os.pardir, os.pardir, 'data', 'raw')
    interim_path = os.path.join(current_dir, os.pardir, os.pardir, os.pardir, 'data', 'interim')
    train_path = os.path.join(interim_path, 'train_with_new_aggregations.csv')
    model_path = os.path.join(current_dir, os.pardir, os.pardir, os.pardir, 'models', 'linear_full_aggr.pickle')
    submission_path = os.path.join(current_dir, os.pardir, os.pardir, os.pardir, 'data', 'processed',
                                   'rf100_full_aggr.csv')

    train = pd.read_csv(train_path)
    file_data_provider = FileDataProvider(train_file=train_path,
                                          parameters_file='',
                                          model_file=model_path,
                                          submission_file=submission_path)
    train_transformer = TrainOnlyTransformer()
    params = {
        'task': 'train',
        'boosting': 'goss',
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.01,
        'subsample': 0.98,
        'max_depth': 8,
        'top_rate': 0.9,
        'num_leaves': 80,
        'min_child_weight': 42,
        'other_rate': 0.07,
        'reg_alpha': 9.6,
        'colsample_bytree': 0.5,
        'min_split_gain': 9.8,
        'reg_lambda': 8.2,
        'min_data_in_leaf': 21,
        'verbose': -1,
    }
    estimator = LightGBMEstimator(params, num_boost_round=10000)

    test_path = os.path.join(interim_path, 'test_with_new_aggregations.csv')
    test = pd.read_csv(test_path)

    pipeline = ModelTrainer(train_transformer,estimator, file_data_provider)

    pipeline.train_and_predict(train, test)

