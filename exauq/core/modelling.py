"""Basic objects for expressing emulation of simulators."""

from __future__ import annotations

import abc
import csv
import dataclasses
import functools
import math
from collections.abc import Collection, Mapping, Sequence
from itertools import product
from numbers import Real
from types import GenericAlias
from typing import Any, Callable, Optional, TypeVar, Union

import numpy as np
from numpy.typing import NDArray

import exauq.utilities.validation as validation
from exauq.core.numerics import equal_within_tolerance
from exauq.utilities.csv_db import Path
from exauq.utilities.validation import check_int

OptionalFloatPairs = tuple[Optional[float], Optional[float]]
T = TypeVar("T")
S = TypeVar("S")


class _LevelTaggedMeta(type):
    """Metaclass used to define custom ``isinstance`` behaviour for ``LevelTagged``."""

    def __instancecheck__(cls, obj: Any) -> bool:
        return get_level(obj) is not None


class LevelTagged(metaclass=_LevelTaggedMeta):
    """Represents objects that have a level attached to them.

    This class is not intended to be instantiated directly, but instead exists mainly to
    support type hinting for objects that have a level attached to them (cf. the
    `set_level`, `get_level` and `remove_level` functions).

    `LevelTagged` supports class subscripting, so that it behaves a like a generic type
    for type hinting. (Note however that `LevelTagged` does not derive from
    `typing.Generic`.) For example, if a function `foo` returns an instance of
    ``exauq.core.modelling.Input`` with a level attached to it, this can be type-hint as:

    ```
    from exauq.core.modelling import Input

    def foo() -> LevelTagged[Input]:
        ...

    ```

    `LevelTagged` also provides custom `isinstance` behaviour that can be used to
    determine whether an object has a level attached to it:

    ```
    from exauq.core.modelling import set_level

    x = Input(1, 2, 3)
    assert not isinstance(x, LevelTagged)

    _ = set_level(x, 99)
    assert isinstance(x, LevelTagged)
    ```

    Attributes
    ----------
    level_attr : str
        The name of the attribute used when attaching a level to an object.
    """

    level_attr = "_LevelTagged_level"

    def __class_getitem__(cls, key):
        return GenericAlias(cls, (key,))


def set_level(obj: T, level: int) -> LevelTagged[T]:
    """Assign a level to an object.

    This creates a new attribute in the supplied object and assigns the given level to
    it. The attribute name is given by ``LevelTagged.level_attr``, but should not be
    accessed directly: to retrieve the level, use ``get_level`` and to remove the level
    use ``remove_level``.

    Parameters
    ----------
    obj : T
        The object to assign a level to.
    level : int
        The level to assign.

    Returns
    -------
    LevelTagged[T]
        The same object `obj` but with the given level attached to it (which can be
        retrieved with ``get_level``).

    Raises
    ------
    ValueError
        If an attribute of `obj` would be overwritten by assigning a level to `obj`.
    """

    if not isinstance(level, int):
        raise TypeError(
            f"Expected 'level' to be an integer, but received {type(level)}."
        )
    elif hasattr(obj, LevelTagged.level_attr):
        raise ValueError(
            f"Cannot set a level on argument 'obj' with value {obj} as existing attribute "
            f"'{LevelTagged.level_attr}' would be overwritten."
        )
    else:
        setattr(obj, LevelTagged.level_attr, level)
        return obj


def get_level(obj: Union[LevelTagged[Any], Any]) -> Optional[int]:
    """Get the level attached to an object.

    Parameters
    ----------
    obj : LevelTagged[Any] or Any
        An object, possibly with a level assigned to it.

    Returns
    -------
    Optional[int]
        The level attached to the object, if present, or else ``None``.
    """

    if not hasattr(obj, LevelTagged.level_attr):
        return None
    else:
        return getattr(obj, LevelTagged.level_attr)


def remove_level(obj: Any) -> None:
    """Remove the level assigned to an object, if present.

    If an object has a level assigned to it, the attribute containing the level is deleted
    from the object. If the object does not have a level assigned to it then no action
    is taken.

    Parameters
    ----------
    obj : Any
        An object.
    """
    if isinstance(obj, LevelTagged):
        delattr(obj, LevelTagged.level_attr)

    return None


class _LevelTaggedOld:
    """An object with a level attached to it.

    This class is not intended to be initialised directly, but rather to be used alongside
    other classes in multiple inheritance. It should be put as the **first** class in an
    inheritance hierarchy, in which case it attaches a read-only attribute `level` to the
    object, defining the object's level.

    Exceptions will be raised if the `level` attribute set by a parent class would be
    overridden as a result of deriving from `LevelTagged`; see the _Raises_ section and
    examples for more details.

    Parameters
    ----------
    level : int
        The level to attach to the object.
    *args : tuple
        Additional arguments passed to the initialiser of parent of this class (as
        determined according to Python's method resolution order).
    **kwargs : dict, optional
        Keyword arguments passed to the initialiser of parent of this class (as
        determined according to Python's method resolution order).

    Attributes
    ----------
    level : int
        (Read-only) The level attached to the object.

    Raises
    ------
    TypeError
        When attempting to subclass from this class, or initialise an instance of this
        class, if doing so would involve masking a `level` attribute set by a parent
        class.

    Examples
    --------
    Here we create a version of a class `A` that is tagged with a level. Notice how there
    is no need to define any behaviour in the initialiser of the class `AWithLevel`. When
    creating an instance of `AWithLevel`, the first argument should be the level, while
    the remaining arguments and keyword arguments should be those required by the
    initialiser of class `A`:
    >>> class A:
    ...     def __init__(self, a: int, b=None):
    ...         self.a = a
    ...         self.b = b
    ...     def a_plus(self, x: int) -> int:
    ...         return self.a + x
    ...     def string_of_b(self) -> str:
    ...         return str(self.b)
    ...
    >>> class AWithLevel(LevelTagged, A):
    ...     pass
    ...
    >>> obj = AWithLevel(level=99, a=1, b=3.14)
    >>> (obj.level, obj.a, obj.string_of_b())
    (99, 1, '3.14')

    If we try to derive from a class that defines a (class) attribute `level`, then an
    error is raised:
    >>> class B:
    ...     def level(self) -> str:
    ...         return "101"
    ...
    >>> class BWithLevel(LevelTagged, B):
    ...     pass
    ...
    TypeError: Cannot create class <class '__main__.BWithLevel'>: attribute 'level' set by a parent class would be masked.

    Similarly, deriving from a class that sets an instance attribute called `level` at
    initialisation will result in an error when attempting to create an instance of the
    'level tagged' class:
    >>> class C:
    ...     def __init__(self):
    ...         self.level = 101
    ...
    >>> class CWithLevel(LevelTagged, C):
    ...     pass
    ...
    >>> c = CWithLevel(level=99)
    TypeError: Cannot initialise object of type <class 'exauq.core.modelling.LevelTagged'>: instance attribute 'level' set by a parent class would be masked.
    """

    def __init__(self, level: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not hasattr(self, "level"):
            self.level = level
        else:
            raise TypeError(
                f"Cannot initialise object of type {__class__}: instance attribute 'level' "
                "set by a parent class would be masked."
            )

    def __init_subclass__(cls) -> None:
        if hasattr(super(), "level"):
            raise TypeError(
                f"Cannot create class {cls}: attribute 'level' set by a parent class "
                "would be masked."
            )

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "level" and hasattr(self, "level"):
            raise AttributeError("Cannot modify this instance's 'level' attribute.")
        else:
            super().__setattr__(name, value)


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

        if not isinstance(other, type(self)):
            return False

        # Check if both are Real or both are Sequences, otherwise return False
        if isinstance(self.value, Real) != isinstance(other.value, Real):
            return False
        if isinstance(self.value, Sequence) != isinstance(other.value, Sequence):
            return False

        # Check for None values
        if self.value is None and other.value is None:
            return True
        if self.value is None or other.value is None:
            return False

        # Compare values
        return equal_within_tolerance(self.value, other.value)

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


class _InputWithLevel(_LevelTaggedOld, Input):
    pass


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
    """(Read-only) An input to a simulator."""

    output: Real
    """(Read-only) The output of the simulator at the input."""

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

    @classmethod
    def read_from_csv(
        cls, path: Path, output_col: int = -1, header: bool = False
    ) -> tuple[TrainingDatum, ...]:
        """Read simulator inputs and outputs from a csv file.

        The data from the csv file is parsed into a sequence of TrainingDatum objects,
        with one datum per (non-empty) row in the csv file. By default, the last column is
        assumed to contain the simulator outputs, though an alternative can be specified
        via the `output_col` keyword argument. The remaining columns are assumed to define
        the coordinates of simulator inputs, in the same order in which they appear in the
        csv file. If the csv file contains a header row then this should be specified so
        that it can be skipped when reading.

        While it is expected that the data in the csv will be rectangular (i.e. each row
        contains the same number of columns), csv files with varying numbers of columns in
        each row will be parsed, so long as the `output_col` is valid for each row. (Users
        should take care in this case, as the various ``TrainingDatum`` constructed will
        have simulator inputs of varying dimension.)

        Parameters
        ----------
        path : str or os.PathLike
            The path to a csv file.
        output_col : int, optional
            (Default: -1) The (0-based) index of the column that defines the simulator
            outputs. Negative values count backwards from the end of the list of columns
            (the default Python behaviour). The default value corresponds to the last
            column in each row.
        header : bool, optional
            (Default: False) Whether the csv contains a header row that should be skipped.

        Returns
        -------
        tuple[TrainingDatum, ...]
            The training data read from the csv file.

        Raises
        ------
        AssertionError
            If the training data contains values that cannot be parsed as finite,
            non-missing floats.
        ValueError
            If 'output_col' does not define a valid column index for all rows.
        """
        training_data = []
        with open(path, mode="r", newline="") as csvfile:
            reader = enumerate(csv.reader(csvfile))
            if header:
                # Skip header, or return empty tuple if file is empty
                try:
                    _ = next(reader)
                except StopIteration:
                    return tuple()

            for i, row in ((i, row) for i, row in reader if len(row) > 0):
                try:
                    parsed_row = cls._parse_csv_row(row)
                except AssertionError as e:
                    raise AssertionError(f"Could not read data from {path}: {e}.")

                try:
                    output = parsed_row.pop(output_col)
                    training_data.append(TrainingDatum(Input(*parsed_row), output))
                except IndexError:
                    raise ValueError(
                        f"'output_col={output_col}' does not define a valid column index for "
                        f"csv data with {len(row)} columns in row {i}."
                    )
                except ValueError:
                    raise AssertionError(
                        f"Could not read data from {path}: infinite or NaN values found."
                    )

        return tuple(training_data)

    @classmethod
    def _parse_csv_row(cls, row: Sequence[str]) -> list[float]:
        try:
            return list(map(float, row))
        except ValueError:
            bad_data = cls._find_first_non_float(row)
            raise AssertionError(f"unable to parse value '{bad_data}' as a float")

    @staticmethod
    def _find_first_non_float(strings: Sequence[str]) -> Optional[str]:
        for s in strings:
            try:
                _ = float(s)
            except ValueError:
                return s

        return None

    def __str__(self) -> str:
        return f"({str(self.input)}, {str(self.output)})"


@dataclasses.dataclass(frozen=True)
class Prediction:
    """Represents the prediction of an emulator at a simulator input.

    The prediction consists of a predicted value together with the variance of the
    prediction, which gives a measure of the uncertainty in the prediction. The standard
    deviation is also provided, as the square root of the variance.

    Two predictions are considered equal if their estimated values and variances agree, to
    within the standard tolerance `exauq.core.numerics.FLOAT_TOLERANCE` as defined by the
    default parameters for `exauq.core.numerics.equal_within_tolerance`.


    Parameters
    ----------
    estimate : numbers.Real
        The estimated value of the prediction.
    variance : numbers.Real
        The variance of the prediction.

    Attributes
    ----------
    estimate : numbers.Real
        (Read-only) The estimated value of the prediction.
    variance : numbers.Real
        (Read-only) The variance of the prediction.
    standard_deviation : numbers.Real
        (Read-only) The standard deviation of the prediction, calculated as the square
        root of the variance.

    See Also
    --------
    ``exauq.core.numerics.equal_within_tolerance`` : Equality up to tolerances.
    """

    estimate: Real
    """(Read-only) The estimated value of the prediction."""

    variance: Real
    """(Read-only) The variance of the prediction."""

    standard_deviation: Real = dataclasses.field(default=None, init=False)
    """(Read-only) The standard deviation of the prediction, calculated as the square
    root of the variance."""

    def __post_init__(self):
        self._validate_estimate(self.estimate)
        self._validate_variance(self.variance)

        std = math.sqrt(self.variance)
        object.__setattr__(self, "standard_deviation", std)

    @staticmethod
    def _validate_estimate(estimate: Any) -> None:
        """Check that the given estimate defines a real number."""

        validation.check_real(
            estimate,
            TypeError(
                f"Expected 'estimate' to define a real number, but received {type(estimate)} "
                "instead."
            ),
        )

    @staticmethod
    def _validate_variance(variance: Any) -> None:
        """Check that the given estimate defines a non-negative real number."""

        validation.check_real(
            variance,
            TypeError(
                "Expected 'variance' to define a real number, but received "
                f"{type(variance)} instead."
            ),
        )

        if variance < 0:
            raise ValueError(
                f"'variance' must be a non-negative real number, but received {variance}."
            )

    def __eq__(self, other: Any) -> bool:
        """Checks equality with another object up to default tolerances."""

        if type(other) is not type(self):
            return False

        return equal_within_tolerance(
            self.estimate, other.estimate
        ) and equal_within_tolerance(self.variance, other.variance)


class GaussianProcessPrediction(Prediction):
    """Represents a prediction arising from a Gaussian process.

    In addition to the functionality provided by ``Prediction``, instances of this class
    include the method `nes_error` for computing the normalised expected square (NES)
    error at a simulator output, utilising the Gaussian assumption.

    Attributes
    ----------
    estimate : numbers.Real
        The estimated value of the prediction.
    variance : numbers.Real
        The variance of the prediction.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def nes_error(self, observed_output: Real) -> float:
        """Calculate the normalised expected squared (NES) error.

        This is defined as the expectation of the squared error divided by the standard
        deviation of the squared error, for a given observed simulation output.
        Mathematically, the denominator of this fraction is zero precisely when this
        prediction's variance is zero; in this case, the NES error is defined to be zero
        if the observed output is equal to this prediction's estimate and ``inf``
        otherwise. However, the implementation of this method checks for whether the
        numerator (i.e. squared error) and/or the denominator (i.e. the standard deviation
        of the squared error) are zero; furthermore, these are done with exact equality
        checks on the floating point numbers involved, rather than a check up to some
        numerical tolerance.

        Parameters
        ----------
        observed_output : Real
            The output of a simulator to compare this prediction with. Must be a finite
            number.

        Returns
        -------
        float
            The normalised expected squared error for this prediction at the given
            simulator output.

        Notes
        -----

        For Gaussian process emulators, the NES error can be computed from the predictive
        variance and squared error of the emulator's prediction at the simulator input:

        ```
        sq_error = (m - observed_output) ** 2
        expected_sq_error = var + sq_error
        std_sq_error = sqrt((2 * (var**2) + 4 * var * sq_error)
        nes_error = expected_sq_error / std_sq_error
        ```

        where `m` is the point estimate of the Gaussian process prediction at `x` and
        `var` is the predictive variance of this estimate.[1]_

        References
        ----------
        .. [1] Mohammadi, H. et al. (2022) "Cross-Validation-based Adaptive Sampling for
           Gaussian process models". DOI: https://doi.org/10.1137/21M1404260
        """

        validation.check_real(
            observed_output,
            TypeError(
                f"Expected 'observed_output' to be of type {Real} but received type "
                f"{type(observed_output)}."
            ),
        )

        validation.check_finite(
            observed_output,
            ValueError(
                f"'observed_output' must be a finite real number, but received {observed_output}."
            ),
        )

        square_err = (self.estimate - observed_output) ** 2
        expected_sq_err = self.variance + square_err
        standard_deviation_sq_err = math.sqrt(
            2 * (self.variance**2) + 4 * self.variance * square_err
        )
        try:
            return float(expected_sq_err / standard_deviation_sq_err)
        except ZeroDivisionError:
            return 0 if expected_sq_err == 0 else float("inf")


class AbstractEmulator(abc.ABC):
    """Represents an abstract emulator for simulators.

    Classes that inherit from this abstract base class define emulators which
    can be trained with simulator outputs using an experimental design
    methodology.
    """

    @property
    @abc.abstractmethod
    def training_data(self) -> tuple[TrainingDatum]:
        """(Read-only) The data on which the emulator has been trained."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def fit_hyperparameters(self) -> Optional[AbstractHyperparameters]:
        """(Read-only) The hyperparameters of the fit for this emulator, or ``None`` if
        this emulator has not been fitted to data."""
        raise NotImplementedError

    @abc.abstractmethod
    def fit(
        self,
        training_data: Collection[TrainingDatum],
        hyperparameters: Optional[AbstractHyperparameters] = None,
        hyperparameter_bounds: Optional[Sequence[OptionalFloatPairs]] = None,
    ) -> None:
        """Fit the emulator to data.

        By default, hyperparameters should be estimated when fitting the emulator to
        data. Alternatively, a collection of hyperparameters may be supplied to
        use directly as the fitted values. If bounds are supplied for the hyperparameters,
        then estimation of the hyperparameters should respect these bounds.

        Parameters
        ----------
        training_data : collection of TrainingDatum
            The pairs of inputs and simulator outputs on which the emulator
            should be trained.
        hyperparameters : AbstractHyperparameters, optional
            (Default: None) Hyperparameters to use directly in fitting the emulator.
            If ``None`` then the hyperparameters should be estimated as part of
            fitting to data.
        hyperparameter_bounds : sequence of tuple[Optional[float], Optional[float]], optional
            (Default: None) A sequence of bounds to apply to hyperparameters
            during estimation, of the form ``(lower_bound, upper_bound)``. All
            but the last tuple should represent bounds for the correlation
            length scale parameters, in the same order as the ordering of the
            corresponding input coordinates, while the last tuple should
            represent bounds for the process variance.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, x: Input) -> Prediction:
        """Make a prediction of a simulator output for a given input.

        Parameters
        ----------
        x : Input
            A simulator input.

        Returns
        -------
        Prediction
            The emulator's prediction of the simulator output from the given input.
        """

        raise NotImplementedError


class AbstractGaussianProcess(AbstractEmulator, metaclass=abc.ABCMeta):
    """Represents an abstract Gaussian process emulator for simulators.

    Classes that inherit from this abstract base class define emulators which
    are implemented as Gaussian process. They should utilise
    `GaussianProcessHyperparameters` for methods and properties that use parameters, or
    return objects, of type `AbstractHyperparameters`.

    Notes
    -----
    The mathematical assumption of being a Gaussian process gives computational benefits,
    such as an explicit formula for calculating the normalised expected squared error at a
    simulator input/output pair.
    """

    @property
    @abc.abstractmethod
    def fit_hyperparameters(self) -> Optional[GaussianProcessHyperparameters]:
        """(Read-only) The hyperparameters of the fit for this Gaussian process emulator,
        or ``None`` if this emulator has not been fitted to data."""
        raise NotImplementedError

    @abc.abstractmethod
    def fit(
        self,
        training_data: Collection[TrainingDatum],
        hyperparameters: Optional[GaussianProcessHyperparameters] = None,
        hyperparameter_bounds: Optional[Sequence[OptionalFloatPairs]] = None,
    ) -> None:
        """Fit the Guassian process emulator to data.

        By default, hyperparameters should be estimated when fitting the Guassian process
        to data. Alternatively, a collection of hyperparameters may be supplied to
        use directly as the fitted values. If bounds are supplied for the hyperparameters,
        then estimation of the hyperparameters should respect these bounds.

        Parameters
        ----------
        training_data : collection of TrainingDatum
            The pairs of inputs and simulator outputs on which the Gaussian process
            should be trained.
        hyperparameters : GaussianProcessHyperparameters, optional
            (Default: None) Hyperparameters for a Gaussian process to use directly in
            fitting the emulator. If ``None`` then the hyperparameters should be estimated
            as part of fitting to data.
        hyperparameter_bounds : sequence of tuple[Optional[float], Optional[float]], optional
            (Default: None) A sequence of bounds to apply to hyperparameters
            during estimation, of the form ``(lower_bound, upper_bound)``. All
            but the last tuple should represent bounds for the correlation
            length scale parameters, in the same order as the ordering of the
            corresponding input coordinates, while the last tuple should
            represent bounds for the process variance.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, x: Input) -> GaussianProcessPrediction:
        """Make a prediction of a simulator output for a given input.

        Parameters
        ----------
        x : Input
            A simulator input.

        Returns
        -------
        GaussianProcessPrediction
            This Gaussian process's prediction of the simulator output from the given
            input.
        """

        raise NotImplementedError

    def covariance_matrix(self, inputs: Sequence[Input]) -> NDArray:
        """Compute the covariance matrix for a sequence of simulator inputs.

        In pseudocode, the covariance matrix for a given collection `inputs` of simulator
        inputs is defined in terms of the correlation matrix as ``sigma^2 *
        correlation(inputs, training_inputs)``, where ``sigma^2`` is the process variance
        for this Gaussian process (which was determined or supplied during training) and
        ``training_inputs`` are the simulator inputs used in training. The only exceptions
        to this are when the supplied `inputs` is empty or if this emulator hasn't been
        trained on data: in these cases an empty array should be returned.

        The default implementation of this method calls the `correlation` method with the
        simulator inputs used for training and the given `inputs`. There is no additional
        error handling, so users requiring error handling should override this method.

        Parameters
        ----------
        inputs : Sequence[Input]
            A sequence of simulator inputs.

        Returns
        -------
        numpy.ndarray
            The covariance matrix for the sequence of inputs, as an array of shape
            ``(len(inputs), n)`` where ``n`` is the number of training data points for
            this Gaussian process.
        """

        if not self.training_data:
            return np.array([])

        training_inputs = tuple(datum.input for datum in self.training_data)
        return self.fit_hyperparameters.process_var * self.correlation(
            inputs, training_inputs
        )

    @abc.abstractmethod
    def correlation(
        self, inputs1: Sequence[Input], inputs2: Sequence[Input]
    ) -> NDArray:
        """Compute the correlation matrix for two sequences of simulator inputs.

        If ``corr_matrix`` is the Numpy array output by this method, the its should be a
        2-dimensional array of shape ``(len(inputs1), len(inputs2))`` such that
        ``corr_matrix[i, j]`` is equal to the correlation between ``inputs1[i]`` and
        ``inputs2[j]`` (or, in pseudocode, ``corr_matrix[i, j] = correlation(inputs1[i],
        inputs2[j])``). The only exception to this is when either of the sequence of
        inputs is empty, in which case an empty array should be returned.

        Parameters
        ----------
        inputs1, inputs2 : Sequence[Input]
            Sequences of simulator inputs.

        Returns
        -------
        numpy.ndarray
            The correlation matrix for the two sequences of inputs, as an array of shape
            ``(len(inputs1), len(inputs2))``.
        """

        raise NotImplementedError


class MultiLevel(dict[int, T]):
    """A multi-level collection of objects, as a mapping from level to objects.

    Objects from this class are `dict` instances that have integer keys. The keys
    should be integers that define the levels, with the value at a key giving the object
    at the corresponding level.

    The only methods of `dict` that this class overrides are those concerning equality
    testing and the result of applying `repr`. An instance of this class is equal to
    another object precisely when the other object is also an instance of this class and
    there is equality as dicts.

    Parameters
    ----------
    elements : Mapping[int, T] or Sequence[T]
        Either a mapping of integers to objects, or another sequence of objects. If a
        sequence is provided that isn't a mapping, then the returned multi-level
        collection will have levels enumerating the objects from the sequence in order,
        starting at level 1.

    Attributes
    ----------
    levels : tuple of int
        (Read-only) The levels in the collection, in increasing order.

    Notes
    -----
    No checks are performed to ensure all objects in the collection have the same type,
    however this class supports type hinting as a generic. Users requiring such checks
    should create a subclass where this is performed.

    Examples
    --------
    Create a multi-level collection of strings:
    >>> ml: MultiLevel[str]
    >>> d = {2: "the", 4: "quick", 6: "brown", 8: "fox"}
    >>> ml = MultiLevel(d)
    >>> ml.levels
    (2, 4, 6, 8)
    >>> ml[2]
    'the'
    >>> ml[8]
    'fox'

    Alternatively, a sequence of elements can be provided, in which case the levels will
    enumerate the sequence in order:
    >>> words = ["the", "quick", "brown", "fox"]
    >>> ml = MultiLevel(d)
    >>> ml.levels
    (1, 2, 3, 4)
    >>> ml[1]
    'the'
    >>> ml[4]
    'fox'

    Note that a MultiLevel collection is not equal to another mapping if the other object
    is not also a MultiLevel instance:
    >>> d = dict(ml)
    >>> ml == d
    False
    >>> dict.__eq__(d, ml)  # equal when considered as dicts
    True
    >>> ml == MultiLevel(d)
    True
    """

    def __init__(self, elements: Union[Mapping[int, T], Sequence[T]]):
        super().__init__(self._parse_elements(elements))

    def _parse_elements(self, elements: Any) -> dict[int, Any]:
        if isinstance(elements, Mapping):
            d = dict(elements)
            if invalid_keys := [k for k in d.keys() if not isinstance(k, int)]:
                key = invalid_keys[0]
                raise ValueError(
                    f"Key '{key}' of invalid type {type(key)} found: keys should be integers "
                    "that define levels."
                )
            else:
                return d
        elif isinstance(elements, Sequence):
            return {i + 1: elem for i, elem in enumerate(elements)}
        else:
            raise TypeError(
                "Argument 'elements' must be a mapping with int keys or a sequence, "
                f"but received object of type {type(elements)}."
            )

    @property
    def levels(self) -> tuple[int, ...]:
        """(Read-only) The levels in the collection, in increasing order."""

        return tuple(sorted(self.keys()))

    def __repr__(self):
        return f"{__class__.__name__}({super().__repr__()})"

    def __eq__(self, other):
        return isinstance(other, __class__) and super().__eq__(other)

    def __ne__(self, other):
        return not self == other

    def map(self, f: Callable[[int, T], S]) -> MultiLevel[S]:
        """Apply a function level-wise.

        Creates a new multi-level collection by applying the given function to each
        (level, value) mapping in this object.

        Parameters
        ----------
        f : Callable[[int, T], S]
            The function to apply to (level, value) pairs.

        Returns
        -------
        MultiLevel[S]
            A new multi-level collection, with levels equal to `self.levels` and
            values created by applying `f` to the (level, value) pairs of `self`.
        """

        return __class__({level: f(level, val) for level, val in self.items()})


def _can_instantiate_multi_level(elements: Any, tp: type) -> bool:
    """Whether a collection of elements can be used to construct a `MultiLevel` instance
    of objects of type `tp`."""

    if isinstance(elements, Sequence) and all(
        isinstance(coeff, tp) for coeff in elements
    ):
        return True
    elif isinstance(elements, Mapping) and all(
        isinstance(k, int) and isinstance(v, tp) for k, v in elements.items()
    ):
        return True
    else:
        return False


class MultiLevelGaussianProcess(MultiLevel[AbstractGaussianProcess], AbstractEmulator):
    """A multi-level Gaussian process (GP) emulator for simulators.

    A multi-level GP is a weighted sum of Gaussian processes, where each GP in the sum is
    considered to be at a particular integer level, and each level only has one GP.
    Training the multi-level GP consists of training each GP independently of the others,
    by supplying training data for specific levels. A key assumption of this class is that
    the constituent GPs are independent of each other, in the probabilistic sense. This
    assumption is used when making predictions for the overall multi-level GP at simulator
    inputs (see the `predict` method for details).

    Parameters
    ----------
    gps : Mapping[int, AbstractGaussianProcess] or Sequence[AbstractGaussianProcess]
        The Gaussian processes for each level in this multi-level GP. If provided as a
        mapping of integers to Gaussian processes, then the levels for this multi-level
        GP will be the keys of this mapping (note these don't need to be sequential or
        start from 1). If provided as a sequence of Gaussian processes, then these will
        be assigned to levels 1, 2, ... in the order provided by the sequence.
    coefficients : Mapping[int, Real] or Sequence[Real] or Real, optional
        (Default: 1) The coefficients to multiply the Gaussian processes at each level by,
        when considering this multi-level GP as a weighted sum of the Gaussian processes.
        If provided as a mapping of integers to real numbers, then the keys will be
        considered as levels, and there must be a coefficient supplied for each level
        defined by `gps` (coefficients for extra levels are ignored). If provided as a
        sequence of real numbers, then the length of the sequence must be equal to the
        number of levels defined by `gps`, in which case the coefficients will be assigned
        to the levels in ascending order, as defined by the ordering of the coefficient
        sequence. If provided as a single real number then this coefficient is assigned to
        each level defined by `gps`.
    """

    def __init__(
        self,
        gps: Union[
            Mapping[int, AbstractGaussianProcess], Sequence[AbstractGaussianProcess]
        ],
        coefficients: Union[Mapping[int, Real], Sequence[Real], Real] = 1,
    ):
        super().__init__(self._parse_gps(gps))
        self._coefficients = self._parse_coefficients(coefficients)

    @staticmethod
    def _parse_gps(
        gps,
    ) -> Union[
        Mapping[Any, AbstractGaussianProcess], Sequence[AbstractGaussianProcess]
    ]:
        """Validate a collection of Gaussian processes for constructing a multi-level GP,
        returning as a multi-level collection or else raising an exception."""

        if not _can_instantiate_multi_level(gps, AbstractGaussianProcess):
            raise TypeError(
                "Expected 'gps' to be a mapping of integers to "
                f"{AbstractGaussianProcess.__name__} or a sequence of "
                f"{AbstractGaussianProcess.__name__}, but received object of "
                f"type {type(gps)} instead."
            )
        else:
            return gps

    def _parse_coefficients(self, coefficients) -> MultiLevel[float]:
        """Validate a collection of coefficients for constructing a multi-level GP,
        returning as a multi-level collection with appropriate levels or else raising
        an exception."""

        if not isinstance(coefficients, Real) and not _can_instantiate_multi_level(
            coefficients, Real
        ):
            raise TypeError(
                "Expected 'coefficients' to be a mapping of integers to real numbers, "
                "a sequence of real numbers, or a real number."
            )
        elif isinstance(coefficients, Real):
            return self._fill_out(float(coefficients), self.levels)
        elif isinstance(coefficients, Sequence) and len(coefficients) != len(self):
            raise ValueError(
                "Expected the same number of coefficients as Gaussian processes (got "
                f"{len(coefficients)} coefficients but expected {len(self)})."
            )
        elif isinstance(coefficients, Sequence):
            coefficients = map(float, coefficients)
            return MultiLevel(dict(zip(self.levels, coefficients)))
        else:
            coefficients = MultiLevel(coefficients).map(
                lambda level, coeff: float(coeff)
            )
            if missing_levels := (set(self.levels) - set(coefficients.levels)):
                missing_levels_str = ", ".join(map(str, sorted(missing_levels)))
                raise ValueError(
                    f"Missing coefficients for levels: {missing_levels_str}."
                )
            else:
                return self._fill_out(coefficients, self.levels)

    @property
    def training_data(self) -> MultiLevel[tuple[TrainingDatum, ...]]:
        """(Read-only) The data on which the Gaussian processes at each level have been
        trained."""

        return self.map(lambda _, gp: gp.training_data)

    @property
    def coefficients(self) -> MultiLevel[Real]:
        """(Read-only) The coefficients that multiply the Gaussian processes level-wise,
        when considering this multi-level GP as a weighted sum of the Gaussian processes
        at each level."""

        return self._coefficients.map(lambda _, coeff: coeff)

    @property
    def fit_hyperparameters(self) -> MultiLevel[Optional[AbstractHyperparameters]]:
        """(Read-only) The hyperparameters of the underlying fitted Gaussian process for
        each level. A value of ``None`` for a level indicates that the Gaussian process
        for the level hasn't been fitted to data."""

        return self.map(lambda _, gp: gp.fit_hyperparameters)

    def fit(
        self,
        training_data: MultiLevel[Collection[TrainingDatum]],
        hyperparameters: Optional[
            Union[
                MultiLevel[GaussianProcessHyperparameters],
                GaussianProcessHyperparameters,
            ]
        ] = None,
        hyperparameter_bounds: Optional[
            Union[
                MultiLevel[Sequence[OptionalFloatPairs]], Sequence[OptionalFloatPairs]
            ]
        ] = None,
    ) -> None:
        """Fit this multi-level Gaussian process to levelled training data.

        The Gaussian process at each level within this multi-level GP is trained on the
        data supplied at the corresponding level. By default, hyperparameters for each
        level's Gaussian process are estimated, although specific hyperparameters can be
        supplied for some or all of the levels to be used when training instead.
        Similarly, bounds on hyperparameter estimation can be supplied for some or all of
        the levels.

        In general, if any of the training data, hyperparameters or bounds contain levels
        not featuring within this multi-level GP, then the data for these extra levels is
        simply ignored.

        Parameters
        ----------
        training_data : MultiLevel[Collection[TrainingDatum]]
            A level-wise collection of pairs of simulator inputs and outputs for training
            the Gaussian processes by level. If data is not supplied for a level featuring
            in `self.levels` then no training is performed at that level.
        hyperparameters : MultiLevel[GaussianProcessHyperparameters] or GaussianProcessHyperparameters, optional
            (Default: None) Either a level-wise collection of hyperparameters to use
            directly when fitting each level's Gaussian process, or a single set of
            hyperparameters to use on each of the levels. If ``None`` then the
            hyperparameters will be estimated at each level when fitting. If a
            ``MultiLevel`` collection is supplied and a level from `self.levels` is
            missing from the collection, then the hyperparameters at that level will be
            estimated when training the corresponding Gaussian process.
        hyperparameter_bounds : MultiLevel[Sequence[OptionalFloatPairs]]] or Sequence[OptionalFloatPairs], optional
            (Default: None) Either a level-wise collection of bounds to apply to
            hyperparameters during estimation, or a single collection of bounds to use on
            each of the levels. If a ``MultiLevel`` collection is supplied and a level
            from `self.levels` is missing from the collection, then the hyperparameters at
            that level will be estimated without any bounding constraints. See the
            documentation for the `fit` method of `AbstractGaussianProcess` for details on
            how bounds should be constructed for each level's Gaussian process.

        See Also
        --------
        AbstractGaussianProcess.fit : Fitting individual Gaussian processes.
        """

        if not isinstance(training_data, MultiLevel):
            raise TypeError(
                f"Expected 'training_data' to be an instance of {MultiLevel.__name__}, but received "
                f"{type(training_data)}."
            )
        else:
            training_data = MultiLevel(
                {
                    level: data
                    for level, data in training_data.items()
                    if level in self.levels
                }
            )

        hyperparameters = self._fill_out(hyperparameters, training_data.levels)
        hyperparameter_bounds = self._fill_out(
            hyperparameter_bounds, training_data.levels
        )
        for level, data in training_data.items():
            try:
                self[level].fit(
                    data,
                    hyperparameters=hyperparameters[level],
                    hyperparameter_bounds=hyperparameter_bounds[level],
                )
            except (TypeError, ValueError) as e:
                raise e.__class__(
                    f"Could not train Gaussian process at level {level}: {e}"
                )

        return None

    @staticmethod
    def _fill_out(
        base: Union[T, MultiLevel[T]], levels: Sequence[int], fill: Optional[T] = None
    ) -> MultiLevel[Optional[T]]:
        """Create a multi-level collection with given levels and filled given objects.

        If `base` is a multi-level collection, then objects at the given `levels` will be
        taken from `base` and any remaining levels will be assigned the value `fill`. If
        `base` is not a multi-level collection then each level in the output collection
        will contain the value `base`.
        """

        if isinstance(base, MultiLevel):
            values = map(lambda level: base[level] if level in base else fill, levels)
            return MultiLevel(dict(zip(levels, values)))
        else:
            return MultiLevel({level: base for level in levels})

    def predict(self, x: Input) -> GaussianProcessPrediction:
        """Predict a simulator output for a given input.

        Parameters
        ----------
        x : Input
            A simulator input.

        Returns
        -------
        GaussianProcessPrediction
            The emulator's prediction of the simulator output from the given the input.

        Notes
        -----
        The prediction for the whole multi-level Gaussian process (GP) is calculated in
        terms of the predictions of the Gaussian processes at each level, together with
        their coefficients in `self.coefficients`, making use of the assumption that the
        GPs at each level are independent of each other. As such, the predicted mean at
        the input `x` is equal to the sum of the predicted means from the level-wise GPs
        multiplied by the corresponding coefficients, while the predicted variance is
        equal to the sum of the predicted variances of the level-wise GPs multiplied by
        the squares of the coefficients.
        """

        if not isinstance(x, Input):
            raise TypeError(
                f"Expected 'x' to be of type {Input.__name__}, but received {type(x)}."
            )

        level_predictions = self.map(lambda level, gp: gp.predict(x))
        estimate = sum(
            p.estimate * self._coefficients[level]
            for level, p in level_predictions.items()
        )
        variance = sum(
            p.variance * (self._coefficients[level] ** 2)
            for level, p in level_predictions.items()
        )
        return GaussianProcessPrediction(estimate, variance)


class AbstractHyperparameters(abc.ABC):
    """A base class for hyperparameters used to train an emulator.

    This class doesn't implement any functionality, but instead is used to indicate to
    type checkers where a class containing hyperparameters for fitting a concrete emulator
    is required. Users should derive from this class when creating concrete classes of
    hyperparameters.
    """

    pass


def _validate_nonnegative_real_domain(arg_name: str):
    """A decorator to be applied to functions with a single real-valued argument called
    `arg_name`. The decorator adds validation that the argument is a real number >= 0.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapped(arg: Real):
            # N.B. Not using try-except here because that would allow single-element Numpy
            # arrays to pass through with deprecation warning.
            if not isinstance(arg, Real):
                raise TypeError(
                    f"Expected '{arg_name}' to be a real number, but received {type(arg)}."
                )

            if arg < 0:
                raise ValueError(f"'{arg_name}' cannot be < 0, but received {arg}.")

            return func(arg)

        return wrapped

    return decorator


@dataclasses.dataclass(frozen=True)
class GaussianProcessHyperparameters(AbstractHyperparameters):
    """Hyperparameters for use in fitting Gaussian processes.

    There are three basic (sets of) hyperparameters used for fitting Gaussian processes:
    correlation length scales, process variance and, optionally, a nugget. These are
    expected to be on a linear scale; tranformation functions for converting to a log
    scale are provided as static methods.

    Equality of `GaussianProcessHyperparameters` objects is tested hyperparameter-wise up
    to the default numerical precision defined in ``exauq.core.numerics.FLOAT_TOLERANCE``
    (see ``exauq.core.numerics.equal_within_tolerance``).

    Parameters
    ----------
    corr_length_scales : sequence or Numpy array of numbers.Real
        The correlation length scale parameters. The length of the sequence or array
        should equal the number of input coordinates for an emulator and each scale
        parameter should be a positive.
    process_var : numbers.Real
        The process variance, which should be positive.
    nugget : numbers.Real, optional
        (Default: None) A nugget, which should be non-negative if provided.

    Attributes
    ----------
    corr_length_scales : sequence or Numpy array of numbers.Real
        (Read-only) The correlation length scale parameters.
    process_var : numbers.Real
        (Read-only) The process variance.
    nugget : numbers.Real, optional
        (Read only, default: None) The nugget, or ``None`` if not supplied.
    """

    corr_length_scales: Union[Sequence[Real], np.ndarray[Real]]
    """(Read-only) The correlation length scale parameters."""

    process_var: Real
    """(Read-only) The process variance."""

    nugget: Optional[Real] = None
    """(Read only, default: None) The nugget, or ``None`` if not supplied."""

    def __post_init__(self):
        if not isinstance(self.corr_length_scales, (Sequence, np.ndarray)):
            raise TypeError(
                "Expected 'corr_length_scales' to be a sequence or Numpy array, but "
                f"received {type(self.corr_length_scales)}."
            )

        nonpositive_corrs = [
            x for x in self.corr_length_scales if not isinstance(x, Real) or x <= 0
        ]
        if nonpositive_corrs:
            nonpositive_element = nonpositive_corrs[0]
            raise ValueError(
                "Expected 'corr_length_scales' to be a sequence or Numpy array of "
                "positive real numbers, but found element "
                f"{nonpositive_element} of type {type(nonpositive_element)}."
            )

        validation.check_real(
            self.process_var,
            TypeError(
                "Expected 'process_var' to be a real number, but received "
                f"{type(self.process_var)}."
            ),
        )
        if self.process_var <= 0:
            raise ValueError(
                "Expected 'process_var' to be a positive real number, but received "
                f"{self.process_var}."
            )

        if self.nugget is not None:
            if not isinstance(self.nugget, Real):
                raise TypeError(
                    f"Expected 'nugget' to be a real number, but received {type(self.nugget)}."
                )

            if self.nugget < 0:
                raise ValueError(
                    f"Expected 'nugget' to be a positive real number, but received {self.nugget}."
                )

    def __eq__(self, other) -> bool:
        if not isinstance(other, self.__class__):
            return False

        try:
            nuggets_equal = (
                self.nugget is None and other.nugget is None
            ) or equal_within_tolerance(self.nugget, other.nugget)
        except TypeError:
            return False

        return all(
            [
                nuggets_equal,
                equal_within_tolerance(
                    self.corr_length_scales, other.corr_length_scales
                ),
                equal_within_tolerance(self.process_var, other.process_var),
            ]
        )

    @staticmethod
    @_validate_nonnegative_real_domain("corr_length_scales")
    def transform_corr(corr_length_scales: Real) -> float:
        """Transform a correlation length scale parameter to a negative log scale.

        This applies the mapping ``corr_length_scale -> -2 * log(corr_length_scale)``,
        using the natural log.
        """
        if corr_length_scales == 0:
            return math.inf

        return -2 * math.log(corr_length_scales)

    @staticmethod
    @_validate_nonnegative_real_domain("process_var")
    def transform_cov(process_var: Real) -> float:
        """Transform a process variance to the (natural) log scale via
        ``process_var -> log(process_var)."""
        if process_var == 0:
            return -math.inf

        return math.log(process_var)

    @staticmethod
    @_validate_nonnegative_real_domain("nugget")
    def transform_nugget(nugget: Real) -> float:
        """Transform a nugget to the (natural) log scale via ``nugget -> log(nugget)``."""
        if nugget == 0:
            return -math.inf

        return math.log(nugget)


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
    bounds : tuple[tuple[Real, Real], ...]
        (Read-only) The bounds defining this domain, as a tuple of pairs of
        real numbers ``((a_1, b_1), ..., (a_n, b_n))``, with each pair ``(a_i, b_i)``
        representing the lower and upper bounds for the corresponding coordinate in the
        domain.

    Parameters
    ----------
    bounds : Sequence[tuple[Real, Real]]
        A sequence of tuples of real numbers ``((a_1, b_1), ..., (a_n, b_n))``, with each
        pair ``(a_i, b_i)`` representing the lower and upper bounds for the corresponding
        coordinate in the domain.

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

    def __init__(self, bounds: Sequence[tuple[Real, Real]]):
        self._validate_bounds(bounds)
        self._bounds = tuple(bounds)
        self._dim = len(bounds)
        self._corners = None

        self._define_corners()

    @staticmethod
    def _validate_bounds(bounds: Sequence[tuple[Real, Real]]) -> None:
        """
        Validates the bounds for initialising the domain of the simulator.

        This method checks that the provided bounds meet the necessary criteria for
        defining a valid n-dimensional rectangular domain. The bounds are expected to be a
        list of tuples, where each tuple represents the lower and upper bounds for a
        coordinate in the domain. The method validates that:

        1. There is at least one dimension provided (the list of bounds is not empty).
        2. Each bound is a tuple of two real numbers.
        3. The lower bound is not greater than the upper bound in any dimension.

        Parameters
        ----------
        bounds : list[tuple[Real, Real]]
            A list of tuples where each tuple represents the bounds for a dimension in the
            domain. Each tuple should contain two real numbers (low, high) where `low` is
            the lower bound and `high` is the upper bound for that dimension.

        Examples
        --------
        This should pass without any issue as the bounds are valid.
        >>> SimulatorDomain.validate_bounds([(0, 1), (0, 1)])

        ValueError: Domain must be at least one-dimensional.
        >>> SimulatorDomain.validate_bounds([])

        ValueError: Each bound must be a tuple of two numbers.
        >>> SimulatorDomain.validate_bounds([(0, 1, 2), (0, 1)])

        TypeError: Bounds must be real numbers.
        >>> SimulatorDomain.validate_bounds([(0, '1'), (0, 1)])

        ValueError: Lower bound cannot be greater than upper bound.
        >>> SimulatorDomain.validate_bounds([(1, 0), (0, 1)])
        """
        if bounds is None:
            raise TypeError("Bounds cannot be None. 'bounds' should be a sequence.")

        if not bounds:
            raise ValueError("At least one pair of bounds must be provided.")

        if not isinstance(bounds, Sequence):
            raise TypeError("Bounds should be a sequence.")

        for bound in bounds:
            if not isinstance(bound, tuple) or len(bound) != 2:
                raise ValueError("Each bound must be a tuple of two numbers.")

            low, high = bound
            if not (isinstance(low, Real) and isinstance(high, Real)):
                raise TypeError("Bounds must be real numbers.")

            if low > high:
                if not equal_within_tolerance(low, high):
                    raise ValueError("Lower bound cannot be greater than upper bound.")

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

    @property
    def bounds(self) -> tuple[tuple[Real, Real], ...]:
        """(Read-only) The bounds defining this domain, as a tuple of pairs of
        real numbers ``((a_1, b_1), ..., (a_n, b_n))``, with each pair ``(a_i, b_i)``
        representing the lower and upper bounds for the corresponding coordinate in the
        domain."""
        return self._bounds

    def scale(self, coordinates: Sequence[Real]) -> Input:
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
        coordinates : collections.abc.Sequence[numbers.Real]
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

    def _within_bounds(self, point: Input) -> bool:
        """
        Check if a single point is within the bounds of the domain.

        Parameters
        ----------
        point : Input
            The point to check.

        Returns
        -------
        bool
            True if the point is within the bounds, False otherwise.
        """
        return all(
            self._bounds[i][0] <= point[i] <= self._bounds[i][1]
            for i in range(self._dim)
        )

    def _validate_points_dim(self, collection: Collection[Input]) -> None:
        """
        Validates that all points in a collection have the same dimensionality as the domain.

        This method checks each point in the provided collection to ensure that it has
        the same number of dimensions (i.e., the same length) as the domain itself. If any
        point in the collection does not have the same dimensionality as the domain, a
        ValueError is raised.

        Parameters
        ----------
        collection : Collection[Input]
            A collection of points (each of type Input) that need to be validated for their
            dimensionality.

        Raises
        ------
        ValueError
            If any point in the collection does not have the same dimensionality as the domain.

        Examples
        --------
        >>> domain = SimulatorDomain([(0, 1), (0, 1)])
        >>> points = [Input(0.5, 0.5), Input(0.2, 0.8)]
        >>> domain._validate_points_dim(points)  # This should pass without any issue

        >>> invalid_points = [Input(0.5, 0.5, 0.3), Input(0.2, 0.8)]
        >>> domain._validate_points_dim(invalid_points)
        ValueError: All points in the collection must have the same dimensionality as the domain.
        """
        if not all(len(point) == self._dim for point in collection):
            raise ValueError(
                "All points in the collection must have the same dimensionality as the domain."
            )

    def _define_corners(self):
        """Generates and returns all the corner points of the domain."""

        unique_corners = []
        for corner in product(*self._bounds):
            input_corner = Input(*corner)
            if input_corner not in unique_corners:
                unique_corners.append(input_corner)

        self._corners = tuple(unique_corners)

    @staticmethod
    def _calculate_distance(point_01: Sequence, point_02: Sequence):
        return sum((c1 - c2) ** 2 for c1, c2 in zip(point_01, point_02)) ** 0.5

    @property
    def get_corners(self) -> tuple[Input]:
        """
        Returns all the corner points of the domain.

        A corner point of a domain is defined as a point where each coordinate
        is equal to either the lower or upper bound of its respective dimension.
        This method calculates all possible combinations of lower and upper bounds
        for each dimension to find all the corner points of the domain.

        Returns
        -------
        tuple[Input]
            A tuple containing all the corner points of the domain. The number of corner
            points is ``2 ** dim``, where dim is the number of dimensions of the domain.

        Examples
        --------
        >>> bounds = [(0, 1), (0, 1)]
        >>> domain = SimulatorDomain(bounds)
        >>> domain.get_corners
        (Input(0, 0), Input(0, 1), Input(1, 0), Input(1, 1))
        """

        return self._corners

    def closest_boundary_points(self, inputs: Collection[Input]) -> tuple[Input]:
        """
        Finds the closest point on the boundary for each point in the input collection.
        Distance is calculated using the Euclidean distance.

        Parameters
        ----------
        inputs : Collection[Input]
            A collection of points for which the closest boundary points are to be found.
            Each point in the collection must be an instance of `Input` and have the same
            dimensionality as the domain.

        Returns
        -------
        tuple[Input]
            The boundary points closest to a point in the given `inputs`.

        Raises
        ------
        ValueError
            If any point in the collection is not within the bounds of the domain.
            If any point in the collection does not have the same dimensionality as the domain.

        Examples
        --------
        >>> bounds = [(0, 1), (0, 1)]
        >>> domain = SimulatorDomain(bounds)
        >>> collection = [Input(0.5, 0.5)]
        >>> domain.closest_boundary_points(collection)
        (Input(0, 0.5), Input(1, 0.5), Input(0.5, 0), Input(0.5, 1))

        Notes
        -----
        The method does not guarantee a unique solution if multiple points on the boundary
        are equidistant from a point in the collection. In such cases, the point that is
        found first in the iteration will be returned.
        """

        # Check if collection is empty
        if not inputs:
            return tuple()

        # Check all points have same dimensionality as domain
        self._validate_points_dim(inputs)

        # Check all points are within domain bounds
        if not all(self._within_bounds(point) for point in inputs):
            raise ValueError(
                "All points in the collection must be within the domain bounds."
            )

        closest_boundary_points = []
        for i in range(self._dim):
            for bound in [self._bounds[i][0], self._bounds[i][1]]:
                min_distance = float("inf")
                closest_point = None
                for point in inputs:
                    if point not in self._corners:
                        modified_point = list(point)
                        modified_point[i] = bound
                        distance = self._calculate_distance(modified_point, point)
                        if distance < min_distance:
                            min_distance = distance
                            closest_point = modified_point

                if closest_point:
                    closest_input = Input(*closest_point)
                    if (
                        closest_input not in closest_boundary_points
                        and closest_input not in self._corners
                    ):
                        closest_boundary_points.append(closest_input)

        return tuple(closest_boundary_points)

    def calculate_pseudopoints(self, inputs: Collection[Input]) -> tuple[Input]:
        """
        Calculates and returns a tuple of pseudopoints for a given collection of input points.

        A pseudopoint in this context is defined as a point on the boundary of the domain,
        or a corner of the domain. This method computes two types of pseudopoints: Boundary
        pseudopoints and Corner pseudopoints, using the `closest_boundary_points` and `get_corners`
        methods respectively.

        Parameters
        ----------
        inputs : Collection[Input]
            A collection of input points for which to calculate the pseudopoints. Each input point
            must have the same number of dimensions as the domain and must lie within the domain's bounds.

        Returns
        -------
        tuple[Input]
            A tuple containing all the calculated pseudopoints.

        Raises
        ------
        ValueError
            If any of the input points have a different number of dimensions than the domain, or if any of the
            input points lie outside the domain's bounds.

        Examples
        --------
        >>> bounds = [(0, 1), (0, 1)]
        >>> domain = SimulatorDomain(bounds)
        >>> inputs = [Input(0.25, 0.25), Input(0.75, 0.75)]
        >>> pseudopoints = domain.calculate_pseudopoints(inputs)
        >>> pseudopoints  # pseudopoints include boundary and corner points
        (Input(0, 0.25), Input(0.25, 0), Input(1, 0.75), Input(0.75, 1), Input(0, 0), Input(0, 1), Input(1, 0), Input(1, 1))
        """

        boundary_points = self.closest_boundary_points(inputs)
        pseudopoints = boundary_points + self._corners

        unique_pseudopoints = []
        for point in pseudopoints:
            if point not in unique_pseudopoints:
                unique_pseudopoints.append(point)
        return tuple(unique_pseudopoints)

    def get_boundary_mesh(self, n: int) -> tuple[Input, ...]:

        check_int(
            n,
            TypeError(f"Expected 'n' to be of type int, but received {type(n)}."),
        )
        if n < 2:
            raise ValueError(
                f"Expected 'n' to be a positive integer >=2 but is equal to {n}."
            )

        mesh_points = []

        # Boundary for each dimension
        boundaries = [np.linspace(*self.bounds[i], n) for i in range(self.dim)]
        
        # Find the points using the product and create for multiple dimensions
        for i in range(self.dim):
            for point in product(*boundaries):

                # Check the point is on the boundary (which it should only do for the dimension bounded)
                if any(self.bounds[i] == point[i]):

                    # Convert the points iteratively into the Input Class
                    input_point = Input(*point)

                    # Check the points are unique
                    if input_point not in mesh_points:

                        # Hence add to list of mesh points
                        mesh_points.append(input_point)

        return tuple(mesh_points)


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

        raise NotImplementedError
