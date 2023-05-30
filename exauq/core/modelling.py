import abc
import dataclasses
from numbers import Real
from typing import (
    Any,
    Union,
    Optional
)
import numpy as np
import exauq.utilities.validation.real as validation


class Input(object):
    """The input to a simulator or emulator.

    Parameters
    ----------
    *args : tuple of numbers.Real
        The coordinates of the input. Each coordinate must define a finite
        number that is not a missing value (i.e. not None or NaN).
    
    Attributes
    ----------
    value : tuple of numbers.Real, numbers.Real or None
        Represents the point as a tuple of real numbers (dim > 1), a single real
        number (dim = 1) or None (dim = 0). Note that finer-grained typing is
        preserved during construction of an `Input`. See the Examples.

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
        """Check that all arguments define finite real numbers, returning the
        supplied tuple if so or raising an exception if not."""
        
        validation.check_entries_not_none(
            args, TypeError("Cannot supply None as an argument")
        )
        validation.check_entries_real(
            args, TypeError('Arguments must be instances of real numbers')
        )
        validation.check_entries_finite(
            args,
            ValueError("Cannot supply NaN or non-finite numbers as arguments")
        )
        
        return args

    @classmethod
    def from_array(cls, input: np.ndarray) -> "Input":
        """Create a simulator input from a Numpy array. 

        Parameters
        ----------
        input : numpy.ndarray
            A 1-dimensional Numpy array defining the coordinates of the input.
            Each array entry should define a finite number that is not a missing
            value (i.e. not None or NaN).

        Returns
        -------
        Input
            A simulator input with coordinates defined by the supplied array.
        """
        
        if not isinstance(input, np.ndarray):
            raise TypeError("'input' must be a Numpy ndarray")

        if not input.ndim == 1:
            raise ValueError("'input' must be a 1-dimensional Numpy array")

        validation.check_entries_not_none(
            input, ValueError("'input' cannot contain None")
        )
        validation.check_entries_real(
            input, ValueError("'input' must be a Numpy array of real numbers")
        )
        validation.check_entries_finite(
            input,
            ValueError("'input' cannot contain NaN or non-finite numbers")
        )

        return cls(*tuple(input))

    def __str__(self) -> str:
        if self._value is None:
            return "()"
        
        return str(self._value)
    
    def __repr__(self) -> str:
        if self._value is None:
            return "Input()"
        
        elif isinstance(self._value, Real):
            return f"Input({repr(self._value)})"
        
        else:
            return f"Input{repr(self._value)}"

    def __eq__(self, other) -> bool:
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
        The output of the simulator at the input. This must be a finite
        number that is not a missing value (i.e. not None or NaN).
    
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
        self._validate_output(self.output)

    @staticmethod
    def _validate_input(input: Any) -> None:
        """Check that an object is an instance of an Input, raising a
        TypeError if not."""

        if not isinstance(input, Input):
            raise TypeError("Argument `input` must be of type Input")
    
    @staticmethod
    def _validate_output(observation: Any) -> None:
        """Check that an object defines a finite real number, raising exceptions
        if not."""
        
        validation.check_not_none(
            observation,
            TypeError("Argument 'output' cannot be None")
        )
        validation.check_real(
            observation,
            TypeError("Argument `output` must define a real number")
        )
        validation.check_finite(
            observation,
            ValueError("Argument 'output' cannot be NaN or non-finite")
        )

    @classmethod
    def list_from_arrays(cls, inputs: np.ndarray, outputs: np.ndarray) -> list["TrainingDatum"]:
        """Create a list of training data from Numpy arrays.

        It is common when working with Numpy for staistical modelling to
        represent a set of `inputs` and corresponding `outputs` with two arrays:
        a 2-dimensional array of inputs (with a row for each input) and a
        1-dimensional array of outputs, where the length of the `outputs` array
        is equal to the length of the first dimension of the `inputs` array.
        This method is a convenience for creating a list of TrainingDatum
        objects from these arrays.

        Parameters
        ----------
        inputs : np.ndarray
            A 2-dimensional array of simulator inputs, with each row defining
            a single input. Thus, the shape of `inputs` is ``(n, d)`` where
            ``n`` is the number of inputs and ``d`` is the number of input
            coordinates.
        outputs : np.ndarray
            A 1-dimensional array of simulator outputs, whose length is equal
            to ``n``, the number of inputs (i.e. rows) in `inputs`. The
            ``i``th entry of `outputs` corresponds to the input at row ``i`` of
            `inputs`.

        Returns
        -------
        TrainingDatum
            A list of training data, created by binding the inputs and
            corresponding outputs together.
        """
        
        return [cls(Input.from_array(input), output)
                for input, output in zip(inputs, outputs)]

    def __str__(self) -> str:      
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
        """(Read-only) Get the data on which the emulator has been, or will be,
        trained."""
        
        return self._training_data
    
    @abc.abstractmethod
    def fit(self, training_data: Optional[list[TrainingDatum]] = None) -> None:
        """Train the emulator on pairs of inputs and simulator outputs.

        If no training data is supplied, then the emulator will be trained on
        the training data currently stored in this object's `training_data`
        property.

        Parameters
        ----------
        training_data : list[TrainingDatum], optional
            (Default: None) A collection of inputs with simulator outputs.
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
