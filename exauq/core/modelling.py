import abc
import dataclasses
import typing


class Experiment(object):
    """The input to a simulator or emulator.

    Parameters
    ----------
    *args : tuple of floats
        The coordinates of the input.
    
    Attributes
    ----------
    value : tuple of floats, float or None
        Represents the point as a tuple (dim > 1), float (dim = 1) or None
        (dim = 0).
    """
    def __init__(self, *args):
        self._value = self._unpack_args(args)

    @staticmethod
    def _unpack_args(args: tuple) -> typing.Union[tuple[typing.Any], typing.Any, None]:
        """Return items from a sequence of arguments, simplifying where
        possible.

        Examples
        --------
        >>> x = Experiment()
        >>> x._unpack_args((1, 2, 3))
        (1, 2, 3)
        
        Single arguments get simplified:
        >>> x._unpack_args(('a'))
        'a'

        Empty argument list returns None:
        >>> x._unpack_args(())
        None
        """
        if len(args) > 1:
            return args
        elif len(args) == 1:
            return args[0]
        else:
            return None

    def __eq__(self, other):
        return type(other) == type(self) and self._value == other.value
    
    @property
    def value(self) -> typing.Union[tuple[float, ...], float, None]:
        """(Read-only) Gets the value of the experiment, as a tuple of floats
        (dim > 1), a float (dim = 1), or None (dim = 0)."""
        return self._value


@dataclasses.dataclass(frozen=True)
class TrainingDatum(object):
    """A training point for an emulator.
    
    Emulators are trained on collections ``(x, f(x))`` where ``x`` is an input
    to a simulator (i.e. an experiment) and ``f(x)`` is the output of the
    simulator ``f`` at ``x`` (i.e. an observation). This dataclass represents
    such pairs of experiments with observations.

    Parameters
    ----------
    experiment : Experiment
        An input to a simulator.
    observations : float
        The output of the simulator at the experiment.
    
    Attributes
    ----------
    experiment : Experiment
        (Read-only) An input to a simulator.
    observations : float
        (Read-only) The output of the simulator at the experiment.
    """
    experiment: Experiment
    observation: float


class AbstractEmulator(abc.ABC):
    """Represents an abstract emulator for simulators.

    Classes that inherit from this abstract base class define emulators which
    can be trained with simulator outputs using an experimental design
    methodology.
 
    Attributes
    ----------
    training_data: list[TrainingDatum] or None
        Defines the pairs of experiments and observations on which this emulator
        has been trained. Each `TrainingDatum` should have a 1-dim
        `Experiment`.
    """
    
    def __init__(self):
        self._training_data = None

    @property
    @abc.abstractmethod
    def training_data(self) -> list[TrainingDatum]:
        """(Read-only) Get the data on which the emulator has been trained."""
        return self._training_data
    
    @abc.abstractmethod
    def fit(self, training_data: list[TrainingDatum]) -> None:
        """Train an emulator on pairs of experiments and observations.

        Parameters
        ----------
        training_data : list[TrainingDatum]
            A collection of experiments with simulator outputs.
        """
        pass


class AbstractSimulator(abc.ABC):
    """Represents an abstract simulator.

    Classes that inherit from this abstract base class define simulators, which
    typically represent programs for calculating the outputs of complex models
    for given experiments (i.e. simulator inputs).
    """
    
    @abc.abstractmethod
    def compute(self, x: Experiment) -> float:
        """Compute the value of this simulator at an experiment.

        Parameters
        ----------
        x : Experiment
            An experiment to evaluate the simulator at.

        Returns
        -------
        float
            The output of the simulator at the experiment `x`.
        """
        pass
