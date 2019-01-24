from abc import ABC, abstractmethod


class ABCParameterSeeker(ABC):

    @abstractmethod
    def search(self,data, labels):
        ...


