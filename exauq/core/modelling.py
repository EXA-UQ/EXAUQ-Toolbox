"""Basic objects for expressing emulation of simulators."""
from __future__ import annotations

import abc
import dataclasses
import math
from collections.abc import Collection, Sequence
from itertools import product
from numbers import Real
from typing import Any, Optional, Union

import numpy as np

import exauq.utilities.validation as validation
from exauq.core.numerics import equal_within_tolerance

OptionalFloatPairs = tuple[Optional[float], Optional[float]]


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
            cls(Input.from_array(input), output) for input, output in zip(inputs, outputs)
        ]

    def __str__(self) -> str:
        return f"({str(self.input)}, {str(self.output)})"


@dataclasses.dataclass(frozen=True)
class Prediction:
    """Represents a predicted value together with the variance of the prediction.

    Two predictions are considered equal if their estimated values and variances agree,
    to within the standard tolerance ``exauq.core.numerics.FLOAT_TOLERANCE`` as defined
    by the default parameters for ``exauq.core.numerics.equal_within_tolerance``.

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

    See Also
    --------
    ``exauq.core.numerics.equal_within_tolerance`` : Equality up to tolerances.
    """

    estimate: Real
    variance: Real

    def __post_init__(self):
        self._validate_estimate(self.estimate)
        self._validate_variance(self.variance)

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

    @abc.abstractmethod
    def fit(
        self,
        training_data: Collection[TrainingDatum],
        hyperparameters: Optional[AbstractHyperparameters] = None,
        hyperparameter_bounds: Optional[Sequence[OptionalFloatPairs]] = None,
    ) -> None:
        """Fit the emulator to data.

        By default, hyperparameters should be estimated when fitting the emulator to
        data. Alternatively, a collection of hyperparamters may be supplied to
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
            length parameters, in the same order as the ordering of the
            corresponding input coordinates, while the last tuple should
            represent bounds for the covariance.
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
            The emulator's prediction of the simulator output from the given the input.
        """

        raise NotImplementedError


class AbstractGaussianProcess(AbstractEmulator, metaclass=abc.ABCMeta):
    """Represents an abstract Gaussian process emulator for simulators.

    Classes that inherit from this abstract base class define emulators which
    are implemented as Gaussian processs. The mathematical assumption of being a Gaussian
    process gives computational benefits, such as an explicit formula for calculating
    the normalised expected squared error at a simulator input/output pair."""

    def nes_error(self, x: Input, observed_output: Real) -> float:
        """Calculate the normalised expected squared (NES) error.

        This is defined as the expectation of the squared error divided by the standard
        deviation of the squared error, at the input and output. If the denominator of
        this fraction is zero, then the NES is defined to be zero if the numerator is also
        zero and ``inf`` otherwise. Note that the check for whether the denominator and
        numerator are zero is done with exact equality checks on the floating point
        numbers involved, rather than a check up to some numerical tolerance.

        Parameters
        ----------
        x : Input
            A simulator input.
        observed_output : Real
            The output of a simulator at `x`.

        Returns
        -------
        float
            The normalised expected squared error for the given simulator input and
            output.

        Raises
        ------
        AssertionError
            If this Gaussian process emulator has not been fit to training data.

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

        assert (
            self.training_data
        ), "Could not compute normalised expected squared error: emulator hasn't been fit to data."

        try:
            prediction = self.predict(x)
        except TypeError:
            raise TypeError(
                f"Expected 'x' to be of type {Input.__name__} but received type {type(x)}."
            ) from None

        square_err = (prediction.estimate - observed_output) ** 2
        expected_sq_err = prediction.variance + square_err
        standard_deviation_sq_err = math.sqrt(
            2 * (prediction.variance**2) + 4 * prediction.variance * square_err
        )
        try:
            return float(expected_sq_err / standard_deviation_sq_err)
        except ZeroDivisionError:
            return 0 if expected_sq_err == 0 else float("inf")


class AbstractHyperparameters(abc.ABC):
    """A base class for hyperparameters used to train an emulator.

    This class doesn't implement any functionality, but instead is used to indicate to
    type checkers where a class containing hyperparameters for fitting a concrete emulator
    is required. Users should derive from this class when creating concrete classes of
    hyperparameters.
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

    Parameters
    ----------
    bounds : Sequence[tuple[Real, Real]]
        A sequence of tuples representing the lower and upper bounds for each coordinate dimension in the domain.

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
        self._bounds = bounds
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
            self._bounds[i][0] <= point[i] <= self._bounds[i][1] for i in range(self._dim)
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
