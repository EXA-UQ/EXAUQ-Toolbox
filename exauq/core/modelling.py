import abc
import dataclasses


class Experiment(object):
    """The input to a simulator or emulator.
    """
    def __init__(self, *args):
        self._value = self._unpack_args(args)

    @staticmethod
    def _unpack_args(args):
        if len(args) > 1:
            return args
        elif len(args) == 1:
            return args[0]
        else:
            return None

    def __eq__(self, other):
        return type(other) == type(self) and self._value == other.value
    
    @property
    def value(self):        
        return self._value


@dataclasses.dataclass(frozen=True)
class TrainingDatum(object):
    experiment: Experiment
    observation: float


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
    
    @abc.abstractmethod
    def compute(self, x: Experiment):
        pass
