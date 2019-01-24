from abc import ABC, abstractmethod


class ABCDataProvider(ABC):

    @abstractmethod
    def load_train(self):
        ...

    @abstractmethod
    def save_parameters(self, parameters):
        ...
