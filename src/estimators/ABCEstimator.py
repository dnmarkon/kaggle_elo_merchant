from abc import ABC, abstractmethod


class ABCEstimator(ABC):

    @abstractmethod
    def fit(self, x, y):
        ...

    @abstractmethod
    def transform(self, x, y):
        ...

    @abstractmethod
    def predict(self, x):
        ...
