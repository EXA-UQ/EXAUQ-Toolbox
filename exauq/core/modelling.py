"""Basic objects for expressing emulation of simulators."""

import abc
import dataclasses
from collections.abc import Sequence
from numbers import Real
from typing import Any, Union

import numpy as np

import exauq.utilities.validation as validation


class Input(Sequence):
    """The input to a simulator or emulator.

    `Input` objects should be thought of as coordinate vectors. They implement the
    Sequence abstract base class from the ``collections.abc module``. Applying the
    ``len`` function to an `Input` object will return the number of coordinates in it.
    Individual coordinates can be extracted from an `Input` by using index
    subscripting (with indexing starting at ``0``).

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

    Empty argument list gives an input with `value` = ``None``:

    >>> x = Input()
    >>> x.value is None
    True

    The ``len`` function provides the number of coordinates:

    >>> len(Input(0.5, 1, 2))
    3

    The individual coordinates and slices of inputs can be retrieved by indexing:

    >>> x = Input(1, 2, 3)
    >>> x[0]
    1
    >>> x[1:]
    Input(2, 3)
    >>> x[-1]
    3
    """

    def __init__(self, *args: Real):
        self._value = self._unpack_args(self._validate_args(args))
        self._dim = len(args)

    @staticmethod
    def _unpack_args(args: tuple[Any, ...]) -> Union[tuple[Any, ...], Any, None]:
        """Return items from a sequence of arguments, simplifying where
        possible.

        Examples
        --------
        >>> x = Input()
        >>> x._unpack_args((1, 2, 3))
        (1, 2, 3)

        Single arguments remain in their tuples:
        >>> x._unpack_args(tuple(['a']))
        ('a',)

        Empty argument list returns None:
        >>> x._unpack_args(()) is None
        True
        """

        if len(args) > 1:
            return args
        elif len(args) == 1:
            return args
        else:
            return None

    @classmethod
    def _validate_args(cls, args: tuple[Any, ...]) -> tuple[Real, ...]:
        """Check that all arguments define finite real numbers, returning the
        supplied tuple if so or raising an exception if not."""

        validation.check_entries_not_none(
            args, TypeError("Input coordinates must be real numbers, not None")
        )
        validation.check_entries_real(
            args, TypeError("Arguments must be instances of real numbers")
        )
        validation.check_entries_finite(
            args, ValueError("Cannot supply NaN or non-finite numbers as arguments")
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
            raise TypeError(
                f"Expected 'input' of type numpy.ndarray but received {type(input)}."
            )

        if not input.ndim == 1:
            raise ValueError(
                "Expected 'input' to be a 1-dimensional numpy.ndarray but received an "
                f"array with {input.ndim} dimensions."
            )

        validation.check_entries_not_none(
            input, ValueError("'input' cannot contain None")
        )
        validation.check_entries_real(
            input, ValueError("'input' must be a numpy.ndarray array of real numbers")
        )
        validation.check_entries_finite(
            input, ValueError("'input' cannot contain NaN or non-finite numbers")
        )

        return cls(*tuple(input))

    def __str__(self) -> str:
        if self._value is None:
            return "()"

        elif self._dim == 1:
            return f"{self._value[0]}"

        return str(self._value)

    def __repr__(self) -> str:
        if self._value is None:
            return "Input()"

        elif self._dim == 1:
            return f"Input({repr(self._value[0])})"

        else:
            return f"Input{repr(self._value)}"

    def __eq__(self, other: Any) -> bool:
        """Returns ``True`` precisely when `other` is an `Input` with the same
        coordinates as this `Input`"""

        return type(other) == type(self) and self.value == other.value

    def __len__(self) -> int:
        """Returns the number of coordinates in this input."""

        return self._dim

    def __getitem__(self, item: Union[int, slice]) -> Union["Input", Real]:
        """Gets the coordinate at the given index of this input, or returns a new
        `Input` built from the given slice of coordinate entries."""

        try:
            subseq = self._value[item]  # could be a single entry or a subsequence
            if isinstance(item, slice):
                return self.__class__(*subseq)

            return subseq

        except TypeError:
            raise TypeError(
                f"Subscript must be an 'int' or slice, but received {type(item)}."
            )

        except IndexError:
            raise IndexError(f"Input index {item} out of range.")

    @property
    def value(self) -> Union[tuple[Real, ...], Real, None]:
        """(Read-only) Gets the value of the input, as a tuple of real
        numbers (dim > 1), a single real number (dim = 1), or None (dim = 0)."""

        if self._value is None:
            return None

        if len(self._value) == 1:
            return self._value[0]

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
            raise TypeError("Argument 'input' must be of type Input")
    
    @staticmethod
    def _validate_output(observation: Any) -> None:
        """Check that an object defines a finite real number, raising exceptions
        if not."""

        validation.check_not_none(
            observation, TypeError("Argument 'output' cannot be None")
        )
        validation.check_real(
            observation, TypeError("Argument 'output' must define a real number")
        )
        validation.check_finite(
            observation, ValueError("Argument 'output' cannot be NaN or non-finite")
        )

    @classmethod
    def list_from_arrays(
        cls, inputs: np.ndarray, outputs: np.ndarray
    ) -> list["TrainingDatum"]:
        """Create a list of training data from Numpy arrays.

        It is common when working with Numpy for statistical modelling to
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

        return [
            cls(Input.from_array(input), output)
            for input, output in zip(inputs, outputs)
        ]

    def __str__(self) -> str:
        return f"({str(self.input)}, {str(self.output)})"


class AbstractEmulator(abc.ABC):
    """Represents an abstract emulator for simulators.

    Classes that inherit from this abstract base class define emulators which
    can be trained with simulator outputs using an experimental design
    methodology.
    """

    @abc.abstractmethod
    def fit(
        self,
        training_data: list[TrainingDatum],
        hyperparameter_bounds: Sequence[tuple[float, float]] = None,
    ) -> None:
        """Train the emulator on pairs of inputs and simulator outputs.

        If bounds are supplied for the hyperparameters, then estimation of the
        hyperparameters should respect these bounds.

        Parameters
        ----------
        training_data : list[TrainingDatum]
            A collection of inputs with simulator outputs.
        hyperparameter_bounds : sequence of tuple[float, float], optional
            (Default: None) A sequence of bounds to apply to hyperparameters
            during estimation, of the form ``(lower_bound, upper_bound)``. All
            but the last tuple should represent bounds for the correlation
            length parameters, in the same order as the ordering of the
            corresponding input coordinates, while the last tuple should
            represent bounds for the covariance.
        """

        pass


class SimulatorDomain(object):
    """
    Class representing the domain of a simulator.

    When considering a simulator as a mathematical function ``f(x)``, the domain is the
    set of all inputs of the function. This class supports domains that are
    n-dimensional rectangles, that is, sets of inputs whose coordinates lie between some
    fixed bounds (which may differ for each coordinate). Membership of a given input
    can be tested using the ``in`` operator; see the examples.

    Attributes
    ----------
    dim : int
        (Read-only) The dimension of this domain, i.e. the number of coordinates inputs
        from this domain have.

    Examples
    --------
    Create a 3-dimensional domain for a simulator with inputs ``(x1, x2, x3)`` where
    ``1 <= x1 <= 2``, ``-1 <= x2 <= 1`` and ``0 <= x3 <= 100``:

    >>> bounds = [(1, 2), (-1, 1), (0, 100)]
    >>> domain = SimulatorDomain(bounds)

    Test whether various inputs lie in the domain:

    >>> Input(1, 0, 100) in domain
    True
    >>> Input(1.5, -1, -1) in domain  # third coordinate outside bounds
    False
    """

    def __init__(self, bounds: list[tuple[Real, Real]]):
        self._bounds = bounds
        self._dim = len(bounds)

    def __contains__(self, item: Any):
        """Returns ``True`` when `item` is an `Input` of the correct dimension and
        whose coordinates lie within the bounds defined by this domain."""
        return (
            isinstance(item, Input)
            and len(item) == self._dim
            and all(
                bound[0] <= item[i] <= bound[1] for i, bound in enumerate(self._bounds)
            )
        )

    @property
    def dim(self) -> int:
        """(Read-only) The dimension of this domain, i.e. the number of coordinates
        inputs from this domain have."""
        return self._dim

    def scale(self, coordinates: Sequence[Real, ...]) -> Input:
        """Scale coordinates from the unit hypercube into coordinates for this domain.

        The unit hypercube is the set of points where each coordinate lies between ``0``
        and ``1`` (inclusive). This method provides a transformation that rescales such
        a point to a point lying in this domain. For each coordinate, if the bounds on
        the coordinate in this domain are ``a_i`` and ``b_i``, then the coordinate
        ``x_i`` lying between ``0`` and ``1`` is transformed to
        ``a_i + x_i * (b_i - a_i)``.

        If the coordinates supplied do not lie in the unit hypercube, then the
        transformation described above will still be applied, in which case the
        transformed coordinates returned will not represent a point within this domain.

        Parameters
        ----------
        coordinates : collections.abc.Sequence[numbers.Real, ...]
            Coordinates of a point lying in a unit hypercube.

        Returns
        -------
        Input
            The coordinates for the transformed point as a simulator input.

        Raises
        ------
        ValueError
            If the number of coordinates supplied in the input argument is not equal
            to the dimension of this domain.

        Examples
        --------
        Each coordinate is transformed according to the bounds supplied to the domain:

        >>> bounds = [(0, 1), (-0.5, 0.5), (1, 11)]
        >>> domain = SimulatorDomain(bounds)
        >>> coordinates = (0.5, 1, 0.7)
        >>> transformed = domain.scale(coordinates)
        >>> transformed
        Input(0.5, 0.5, 8.0)

        The resulting `Input` is contained in the domain:

        >>> transformed in domain
        True
        """

        n_coordinates = len(coordinates)
        if not n_coordinates == self.dim:
            raise ValueError(
                f"Expected 'coordinates' to be a sequence of length {self.dim} but "
                f"received sequence of length {n_coordinates}."
            )

        return Input(
            *map(
                lambda x, bnds: bnds[0] + x * (bnds[1] - bnds[0]),
                coordinates,
                self._bounds,
            )
        )


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
