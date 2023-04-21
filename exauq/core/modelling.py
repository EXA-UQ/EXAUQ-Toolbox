import abc


class AbstractEmulator(abc.ABC):
    """Represents an abstract emulator for simulators.
    """
    
    def __init__(self):
        self._training_data = None

    @property
    @abc.abstractmethod
    def training_data(self):
        return self._training_data
    
    @abc.abstractmethod
    def fit(self, training_data):
        pass


class AbstractSimulator(abc.ABC):
    """Represents an abstract simulator.
    """
    pass
