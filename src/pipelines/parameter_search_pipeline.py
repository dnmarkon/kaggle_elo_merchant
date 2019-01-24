import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from src.data.FileDataProvider import FileDataProvider
from src.pipelines.GridSearchCVSeeker import GridSearchCVSeeker
from src.transformers.TrainOnlyTransformer import TrainOnlyTransformer


class ParameterSearchPipeline:

    def __init__(self,
                 data_provider,
                 transformer,
                 parameter_seeker):
        self._data_provider = data_provider
        self._transformer = transformer
        self._parameter_seeker = parameter_seeker

    def run(self):
        data = self._data_provider.load_train()

        transformed_data, labels = self._transformer.fit_transform(data)

        best_parameters = self._parameter_seeker.search(transformed_data, labels)

        self._data_provider.save_parameters(best_parameters)


if __name__ == '__main__':
    current_dir = os.getcwd()
    root_path = os.path.join(current_dir, os.pardir, os.pardir, 'data', 'raw')
    train_path = os.path.join(root_path, 'train.csv')
    interim_path = os.path.join(current_dir, os.pardir, os.pardir, 'data', 'interim')
    params_path = os.path.join(root_path, 'support_vector_regressor.json')

    file_data_provider = FileDataProvider(train_file=train_path, parameters_file=params_path)
    train_transformer = TrainOnlyTransformer()
    estimator = SVR()
    parameters = {'n_estimators': [10, 20, 50, 100], }
    grid_parameter_seeker = GridSearchCVSeeker(estimator=estimator, parameters=parameters, cv=5)

    pipeline = ParameterSearchPipeline(file_data_provider, train_transformer, grid_parameter_seeker)

    pipeline.run()
