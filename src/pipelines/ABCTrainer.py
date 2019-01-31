from abc import ABC, abstractmethod


class ABCTrainer(ABC):

    @abstractmethod
    def train(self, df):
        ...

    @abstractmethod
    def predict(self, df):
        ...

    @abstractmethod
    def train_and_save(self, df):
        ...

    @abstractmethod
    def train_and_predict(self, train_set, test_set):
        ...

    @abstractmethod
    def cross_validate_and_predict(self, train_set, test_set, cv):
        ...