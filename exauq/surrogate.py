from abc import ABC, abstractmethod


class Surrogate(ABC):
    """
    Surrogate model/emulator
    """
    @abstractmethod
    def fit(self, theta):
        pass

    @abstractmethod
    def predict(self):
        pass
