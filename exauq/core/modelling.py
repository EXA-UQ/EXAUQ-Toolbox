import abc
import dataclasses
from numbers import Real
from typing import (
    Any,
    Union
)


class Experiment(object):
    """The input to a simulator or emulator.

    Parameters
    ----------
    *args : tuple of numbers.Real
        The coordinates of the input.
    
    Attributes
    ----------
    value : tuple of numbers.Real, numbers.Real or None
        Represents the point as a tuple of real numbers (dim > 1), a single real
        number (dim = 1) or None (dim = 0). Note that finer-grained typing is
        preserved during construction of an `Experiment`. See the Examples.
    
    Raises
    ------
    TypeError
        If any of the inputs to the constructor don't define real numbers.

    Examples
    --------
    >>> x = Experiment(1, 2, 3)
    >>> x.value
    (1, 2, 3)
    
    Single arguments just return a number:
    >>> x = Experiment(2.1)
    >>> x.value
    2.1

    Types are preserved coordinate-wise:
    >>> import numpy as np
    >>> x = Experiment(1.3, np.float64(2), np.int16(1))
    >>> print([type(a) for a in x.value])
    [<class 'float'>, <class 'numpy.float64'>, <class 'numpy.int16'>]

    Empty argument list gives an experiment with value = None:
    >>> x = Experiment()
    >>> x.value
    None
    """
    def __init__(self, *args):
        self._value = self._unpack_args(self._check_args_real(args))

    @staticmethod
    def _unpack_args(args: tuple[Any, ...]) -> Union[tuple[Any, ...], Any, None]:
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
    
    @staticmethod
    def _check_args_real(args: tuple[Any, ...]) -> tuple[Real, ...]:
        """Check that all arguments define real numbers, returning the supplied
        tuple if so or raising a TypeError otherwise."""
        if not all(isinstance(x, Real) for x in args):
            raise TypeError('Arguments must be instances of real numbers')
        
        return args

    def __str__(self):
        if self._value is None:
            return "()"
        
        return str(self._value)
    
    def __repr__(self):
        if self._value is None:
            return "Experiment()"
        
        elif isinstance(self._value, Real):
            return f"Experiment({repr(self._value)})"
        
        else:
            return f"Experiment{repr(self._value)}"

    def __eq__(self, other):
        return type(other) == type(self) and self._value == other.value
    
    @property
    def value(self) -> Union[tuple[Real, ...], Real, None]:
        """(Read-only) Gets the value of the experiment, as a tuple of real
        numbers (dim > 1), a single real number (dim = 1), or None (dim = 0)."""
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
    observation : numbers.Real
        The output of the simulator at the experiment.
    
    Attributes
    ----------
    experiment : Experiment
        (Read-only) An input to a simulator.
    observation : numbers.Real
        (Read-only) The output of the simulator at the experiment.
    """
    
    experiment: Experiment
    observation: Real

    def __post_init__(self):
        self._validate_experiment(self.experiment)
        self._validate_real(self.observation)

    @staticmethod
    def _validate_experiment(experiment):
        """Check that an object is an instance of an Experiment, raising a
        TypeError if not."""
        if not isinstance(experiment, Experiment):
            raise TypeError("Argument `experiment` must be of type Experiment")
    
    @staticmethod
    def _validate_real(observation):
        """Check that an object is an instance of a real number, raising a
        TypeError if not."""
        if not isinstance(observation, Real):
            raise TypeError("Argument `observation` must define a real number")

    def __str__(self):
        return f"({str(self.experiment)}, {str(self.observation)})"


class AbstractEmulator(abc.ABC):
    """Represents an abstract emulator for simulators.

    Classes that inherit from this abstract base class define emulators which
    can be trained with simulator outputs using an experimental design
    methodology.
 
    Attributes
    ----------
    training_data: list[TrainingDatum] or None
        Defines the pairs of experiments and observations on which this emulator
        has been trained.
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
    def compute(self, x: Experiment) -> Real:
        """Compute the value of this simulator at an experiment.

        Parameters
        ----------
        x : Experiment
            An experiment to evaluate the simulator at.

        Returns
        -------
        numbers.Real
            The output of the simulator at the experiment `x`.
        """
        pass
