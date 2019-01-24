from abc import ABC, abstractmethod


class ABCTransformer(ABC):

    @abstractmethod
    def fit(self, x):
        ...

    @abstractmethod
    def transform(self, x):
        ...

    @abstractmethod
    def fit_transform(self, x):
        ...


