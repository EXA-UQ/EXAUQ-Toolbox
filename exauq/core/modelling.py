import abc
from collections.abc import Iterable
import dataclasses
from numbers import Real
from typing import (
    Any,
    Union
)
import numpy as np


class Input(object):
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
        preserved during construction of an `Input`. See the Examples.
    
    Raises
    ------
    TypeError
        If any of the inputs to the constructor don't define real numbers.

    Examples
    --------
    >>> x = Input(1, 2, 3)
    >>> x.value
    (1, 2, 3)
    
    Single arguments just return a number:
    >>> x = Input(2.1)
    >>> x.value
    2.1

    Types are preserved coordinate-wise:
    >>> import numpy as np
    >>> x = Input(1.3, np.float64(2), np.int16(1))
    >>> print([type(a) for a in x.value])
    [<class 'float'>, <class 'numpy.float64'>, <class 'numpy.int16'>]

    Empty argument list gives an input with value = None:
    >>> x = Input()
    >>> x.value
    None
    """
    def __init__(self, *args):
        self._value = self._unpack_args(self._validate_args(args))

    @staticmethod
    def _unpack_args(args: tuple[Any, ...]) -> Union[tuple[Any, ...], Any, None]:
        """Return items from a sequence of arguments, simplifying where
        possible.

        Examples
        --------
        >>> x = Input()
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
    
    @classmethod
    def _validate_args(cls, args: tuple[Any, ...]) -> tuple[Real, ...]:
        """Check that all arguments define real numbers, returning the supplied
        tuple if so or raising errors otherwise."""
        
        if not cls._no_none_entries(args):
            raise TypeError("Cannot supply None as an argument")

        if not cls._all_entries_real(args):
            raise TypeError('Arguments must be instances of real numbers')
        
        if not cls._all_entries_finite(args):
            raise ValueError("Cannot supply NaN or non-finite numbers as arguments")

        return args

    @staticmethod
    def _no_none_entries(iter: Iterable):
        return all(x is not None for x in iter)
    
    @staticmethod
    def _all_entries_real(iter: Iterable):
        return all(isinstance(x, Real) for x in iter)

    @staticmethod
    def _all_entries_finite(iter: Iterable):
        return all(np.isfinite(x) for x in iter)

    @classmethod
    def from_array(cls, input: np.ndarray):
        if not isinstance(input, np.ndarray):
            raise TypeError("'input' must be a Numpy ndarray")

        if not input.ndim == 1:
            raise ValueError("'input' must be a 1-dimensional Numpy array")

        if not cls._no_none_entries(input):
            raise ValueError("'input' cannot contain None")

        if not cls._all_entries_real(input):
            raise ValueError("'input' must be a Numpy array of real numbers")
        
        if not cls._all_entries_finite(input):
            raise ValueError("'input' cannot contain NaN or non-finite numbers")

        return cls(*tuple(input))

    def __str__(self):
        if self._value is None:
            return "()"
        
        return str(self._value)
    
    def __repr__(self):
        if self._value is None:
            return "Input()"
        
        elif isinstance(self._value, Real):
            return f"Input({repr(self._value)})"
        
        else:
            return f"Input{repr(self._value)}"

    def __eq__(self, other):
        return type(other) == type(self) and self._value == other.value
    
    @property
    def value(self) -> Union[tuple[Real, ...], Real, None]:
        """(Read-only) Gets the value of the input, as a tuple of real
        numbers (dim > 1), a single real number (dim = 1), or None (dim = 0)."""
        return self._value


@dataclasses.dataclass(frozen=True)
class TrainingDatum(object):
    """A training point for an emulator.
    
    Emulators are trained on collections ``(x, f(x))`` where ``x`` is an input
    to a simulator and ``f(x)`` is the output of the simulator ``f`` at ``x``.
    This dataclass represents such pairs of inputs and simulator outputs.

    Parameters
    ----------
    input : Input
        An input to a simulator.
    output : numbers.Real
        The output of the simulator at the input.
    
    Attributes
    ----------
    input : Input
        (Read-only) An input to a simulator.
    output : numbers.Real
        (Read-only) The output of the simulator at the input.
    """
    
    input: Input
    output: Real

    def __post_init__(self):
        self._validate_input(self.input)
        self._validate_real(self.output)

    @staticmethod
    def _validate_input(input: Any):
        """Check that an object is an instance of an Input, raising a
        TypeError if not."""
        if not isinstance(input, Input):
            raise TypeError("Argument `input` must be of type Input")
    
    @staticmethod
    def _validate_real(observation: Any):
        """Check that an object is an instance of a real number, raising a
        TypeError if not."""
        if not isinstance(observation, Real):
            raise TypeError("Argument `output` must define a real number")

    @classmethod
    def list_from_arrays(cls, inputs: np.ndarray, outputs: np.ndarray):
        return [cls(Input.from_array(input), output)
                for input, output in zip(inputs, outputs)]

    def __str__(self):
        return f"({str(self.input)}, {str(self.output)})"


class AbstractEmulator(abc.ABC):
    """Represents an abstract emulator for simulators.

    Classes that inherit from this abstract base class define emulators which
    can be trained with simulator outputs using an experimental design
    methodology.
 
    Attributes
    ----------
    training_data: list[TrainingDatum] or None
        Defines the pairs of inputs and simulator outputs on which this emulator
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
        """Train an emulator on pairs of inputs and simulator outputs.

        Parameters
        ----------
        training_data : list[TrainingDatum]
            A collection of inputs with simulator outputs.
        """
        pass


class AbstractSimulator(abc.ABC):
    """Represents an abstract simulator.

    Classes that inherit from this abstract base class define simulators, which
    typically represent programs for calculating the outputs of complex models
    for given inputs.
    """
    
    @abc.abstractmethod
    def compute(self, x: Input) -> Real:
        """Compute the value of this simulator at an input.

        Parameters
        ----------
        x : Input
            An input to evaluate the simulator at.

        Returns
        -------
        numbers.Real
            The output of the simulator at the input `x`.
        """
        pass
