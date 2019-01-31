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

        self._data_provider.save_best_parameters(best_parameters)



