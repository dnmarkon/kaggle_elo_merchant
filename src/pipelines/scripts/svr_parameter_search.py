import os

from sklearn.svm import SVR

from src.data.FileDataProvider import FileDataProvider
from src.pipelines.GridSearchCVSeeker import GridSearchCVSeeker
from src.pipelines.ParameterSearchPipeline import ParameterSearchPipeline
from src.transformers.TrainOnlyTransformer import TrainOnlyTransformer

if __name__ == '__main__':
    current_dir = os.getcwd()
    root_path = os.path.join(current_dir, os.pardir, os.pardir, os.pardir, 'data', 'raw')
    train_path = os.path.join(root_path, 'train.csv')
    interim_path = os.path.join(current_dir, os.pardir, os.pardir, os.pardir, 'data', 'interim')
    params_path = os.path.join(root_path, 'support_vector_regressor.json')

    file_data_provider = FileDataProvider(train_file=train_path, parameters_file=params_path, model_file='')
    train_transformer = TrainOnlyTransformer()
    estimator = SVR()
    parameters = {'kernel': ['linear','rbf', 'linear', 'poly'],
                  'gamma': ['auto', 'scale', 2**(-10), 2**(-5), 2, 2**3],
                  'C': [2**(-5), 2**(-3), 2, 2**5, 2**10]}
    grid_parameter_seeker = GridSearchCVSeeker(estimator=estimator, parameters=parameters, cv=3)

    pipeline = ParameterSearchPipeline(file_data_provider, train_transformer, grid_parameter_seeker)

    pipeline.run()
