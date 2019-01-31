from abc import ABC, abstractmethod


class ABCDataProvider(ABC):

    @abstractmethod
    def load_train(self):
        ...

    @abstractmethod
    def save_best_parameters(self, parameters):
        ...

    @abstractmethod
    def save_model(self, model):
        ...

    @abstractmethod
    def load_model(self):
        ...

    @abstractmethod
    def save_submission(self, x_test, predictions):
        ...
