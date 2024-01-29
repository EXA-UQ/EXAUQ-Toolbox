import copy
import math
from collections.abc import Collection
from numbers import Real
from typing import Any, Optional, Type, Union

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from exauq.core.modelling import (
    AbstractEmulator,
    AbstractGaussianProcess,
    Input,
    SimulatorDomain,
    TrainingDatum,
)
from exauq.core.numerics import equal_within_tolerance
from exauq.utilities.optimisation import maximise
from exauq.utilities.validation import check_int


class SimpleDesigner(object):
    """A designer producing simulator inputs based on random generation.

    This designer produces simulator inputs by sampling each coordinate uniformly. The
    inputs created all belong to the supplied simulator domain.

    Parameters
    ----------
    domain : SimulatorDomain
        A domain for a simulator.
    """

    def __init__(self, domain: SimulatorDomain):
        self._domain = domain

    def make_design_batch(self, size: int) -> list[Input]:
        """Create a batch of new simulator inputs.

        The inputs returned are created by sampling each coordinate uniformly.

        Parameters
        ----------
        size : int
            The number of inputs to create.

        Returns
        -------
        list[Input]
            A batch of new simulator inputs.
        """
        check_int(
            size,
            TypeError(f"Expected 'size' to be an integer but received {type(size)}."),
        )
        if size < 0:
            raise ValueError(
                f"Expected 'size' to be a non-negative integer but is equal to {size}."
            )

        rng = np.random.default_rng()
        return [
            self._domain.scale(rng.uniform(size=self._domain.dim)) for _ in range(size)
        ]


class SingleLevelAdaptiveSampler:
    """Single level adaptive sampling (SLAS) for training emulators.

    Implements the cross-validation-based adaptive sampling for emulators, as
    described in Mohammadi et. al. (2022).

    Parameters
    ----------
    initial_data: finite collection of TrainingDatum
        Training data on which the emulator will initially be trained.
    """

    def __init__(self, initial_data: Collection[TrainingDatum]):
        self._initial_data = self._validate_initial_data(initial_data)
        self._esloo_errors = None

    @classmethod
    def _validate_initial_data(cls, initial_data):
        try:
            length = len(initial_data)  # to catch infinite iterators
            if not all([isinstance(x, TrainingDatum) for x in initial_data]):
                raise TypeError

            if length == 0:
                raise ValueError

            return initial_data

        except TypeError:
            raise TypeError(
                f"{cls.__name__} must be initialised with a (finite) collection of "
                "TrainingDatum"
            )

        except ValueError:
            raise ValueError("'initial_data' must be nonempty")

    def __str__(self) -> str:
        return f"SingleLevelAdaptiveSampler designer with initial data {str(self._initial_data)}"

    def __repr__(self) -> str:
        return f"SingleLevelAdaptiveSampler(initial_data={repr(self._initial_data)})"

    def train(self, emulator: AbstractEmulator) -> AbstractEmulator:
        """Train an emulator using the single-level adaptive sampling method.

        This will train the emulator on the initial data that was supplied during
        construction of this object.

        Parameters
        ----------
        emulator : AbstractEmulator
            The emulator to train.

        Returns
        -------
        AbstractEmulator
            A new emulator that has been trained with the initial training data and
            using the SLAS methodology. A new object is returned of the same ``type`` as
            `emulator`.
        """

        return_emulator = copy.copy(emulator)
        return_emulator.fit(self._initial_data)
        return return_emulator

    def make_design_batch(self, emulator: AbstractEmulator, size: int = 1):
        if emulator.training_data:
            self._esloo_errors = [0.5] * len(emulator.training_data)

        return [Input(1)] * size

    @property
    def esloo_errors(self):
        return self._esloo_errors


def compute_loo_errors_gp(
    gp: AbstractGaussianProcess,
    domain: SimulatorDomain,
    gp_for_errors: Optional[AbstractGaussianProcess] = None,
) -> AbstractGaussianProcess:
    """Calculate a Gaussian process trained on normalised expected squared leave-one-out
    (LOO) errors.

    The errors are computed from the supplied Gaussian process, `gp`. This involves
    training a Gaussian process (GP) for each leave-one-out subset of the training data of
    `gp` and calculating the normalised expected squared error at the left-out training
    point for each intermediary GP. Note that the intermediary leave-one-out GPs are fit
    to data using the fitted hyperparameters *from the supplied `gp`*, which avoids costly
    re-estimation of hyperparameters. The resulting errors, together with the
    corresponding left out simulator inputs, form the training data for the output GP. The
    output GP is trained with a lower bound on the correlation length scale parameters
    (see the Notes section).

    By default, the returned ``AbstractGaussianProcess`` object will be a deep copy of
    `gp` trained on the leave-one-out errors. Alternatively, another
    ``AbstractGaussianProcess`` can be supplied that will be trained on the leave-one-out
    errors and returned (thus it will be modified in-place as well as returned).

    Parameters
    ----------
    gp : AbstractGaussianProcess
        A Gaussian process to calculate the normalised expected squared LOO errors for.
    domain : SimulatorDomain
        The domain of a simulator that the Gaussian process `gp` emulates. The data on
        which `gp` is trained are expected to have simulator inputs only from this domain.
    gp_for_errors : Optional[AbstractGaussianProcess], optional
        (Default: None) Another Gaussian process that is trained on the LOO errors to
        create the output to this function. If ``None`` then a deep copy of `gp` will
        be used instead.

    Returns
    -------
    AbstractGaussianProcess
        A Gaussian process that is trained on the normalised expected square LOO errors
        of `gp`. If `gp_for_errors` was supplied then (a reference to) this object will be
        returned (except now it has been fit to the LOO errors).

    Raises
    ------
    ValueError
        If any of the training inputs in `gp` do not belong to the simulator domain `domain`.

    Notes
    -----
    The lower bound on the correlation length scale parameters is obtained by scaling
    the value ``sqrt(-0.5 / log(10 ** (-8)))``, for each dimension of the simulator
    domain, from the unit interval [0, 1] to the interval of the corresponding dimension.
    """

    if not isinstance(gp, AbstractGaussianProcess):
        raise TypeError(
            f"Expected 'gp' to be of type AbstractGaussianProcess, but received {type(gp)} "
            "instead."
        )

    if not isinstance(domain, SimulatorDomain):
        raise TypeError(
            f"Expected 'domain' to be of type SimulatorDomain, but received {type(domain)} "
            "instead."
        )

    if not all(datum.input in domain for datum in gp.training_data):
        raise ValueError(
            "Expected all training inputs in 'gp' to belong to the domain 'domain', but "
            "this is not the case."
        )

    if not (gp_for_errors is None or isinstance(gp_for_errors, AbstractGaussianProcess)):
        raise TypeError(
            "Expected 'gp_for_errors' to be None or of type AbstractGaussianProcess, but "
            f"received {type(gp_for_errors)} instead."
        )

    error_training_data = []
    loo_gp = copy.deepcopy(gp)
    for leave_out_idx, datum in enumerate(gp.training_data):
        # Fit LOO GP, storing into loo_gp
        _ = compute_loo_gp(gp, leave_out_idx, loo_gp=loo_gp)

        # Add training input and nes error
        nes_loo_error = loo_gp.nes_error(datum.input, datum.output)
        error_training_data.append(TrainingDatum(datum.input, nes_loo_error))

    gp_e = gp_for_errors if gp_for_errors is not None else copy.deepcopy(gp)

    # Note: the following is a simplification of sqrt(-0.5 / log(10 ** (-8))) from paper
    CORR_BOUND = 0.25 / math.sqrt(math.log(10))
    bounds = [(bnd, None) for bnd in domain.scale([CORR_BOUND] * domain.dim)] + [
        (None, None)
    ]
    gp_e.fit(error_training_data, hyperparameter_bounds=bounds)
    return gp_e


def compute_loo_gp(
    gp: AbstractGaussianProcess,
    leave_out_idx: int,
    loo_gp: Optional[AbstractGaussianProcess] = None,
) -> AbstractGaussianProcess:
    """Calculate a leave-one-out (LOO) Gaussian process.

    The returned Gaussian process (GP) is obtained by training it on all training data
    from the supplied GP except for one datum (the 'left out' datum). It is trained using
    the fitted hyperparameters *from the supplied Gaussian process `gp`*.

    By default, the returned ``AbstractGaussianProcess`` object will be a deep copy of
    `gp` trained on the leave-one-out data. Alternatively, another
    ``AbstractGaussianProcess`` can be supplied that will be trained on the leave-one-out
    data and returned (thus it will be modified in-place as well as returned). This can
    be more efficient when repeated calculation of a LOO GP is required.

    Parameters
    ----------
    gp : AbstractGaussianProcess
        A Gaussian process to form the basis for the LOO GP.
    leave_out_idx : int
        The index for the training datum of `gp` to leave out. This should be an index
        of the sequence returned by the ``gp.training_data`` property.
    loo_gp : Optional[AbstractGaussianProcess], optional
        (Default: None) Another Gaussian process that is trained on the LOO data and then
        returned. If ``None`` then a deep copy of `gp` will be used instead.

    Returns
    -------
    AbstractGaussianProcess
        A Gaussian process that is trained on all training data from `gp` except the datum
        that is left out.

    Raises
    ------
    ValueError
        If the supplied ``AbstractGaussianProcess`` hasn't been trained on any data.
    """

    if not isinstance(gp, AbstractGaussianProcess):
        raise TypeError(
            f"Expected 'gp' to be of type AbstractGaussianProcess, but received {type(gp)} "
            "instead."
        )

    if not isinstance(leave_out_idx, int):
        raise TypeError(
            f"Expected 'leave_out_idx' to be of type int, but received {type(leave_out_idx)} "
            "instead."
        )

    if not (loo_gp is None or isinstance(loo_gp, AbstractGaussianProcess)):
        raise TypeError(
            "Expected 'loo_gp' to be None or of type AbstractGaussianProcess, but "
            f"received {type(loo_gp)} instead."
        )

    if len(gp.training_data) == 0:
        raise ValueError(
            "Cannot compute leave one out error with 'gp' because it has not been "
            "trained on data."
        )

    if not 0 <= leave_out_idx < len(gp.training_data):
        raise ValueError(
            f"Leave out index {leave_out_idx} is not within the bounds of the training "
            "data for 'gp'."
        )

    remaining_data = (
        gp.training_data[:leave_out_idx] + gp.training_data[leave_out_idx + 1 :]
    )
    loo_gp_ = loo_gp if loo_gp is not None else copy.deepcopy(gp)
    loo_gp_.fit(remaining_data, hyperparameters=gp.fit_hyperparameters)
    return loo_gp_


class PEICalculator:
    """
    A calculator of pseudo-expected improvement (PEI) for Gaussian processes.

    Parameters
    ----------
    domain : SimulatorDomain
        The domain of a simulation.
    gp : AbstractGaussianProcess
        A Gaussian process model.

    Attributes
    ----------
    gp : AbstractGaussianProcess
        Readonly property that returns the Gaussian Process model used in the calculator.
    repulsion_points : tuple
        Readonly property that returns the current set of repulsion points.
    domain : SimulatorDomain
        Readonly property that returns the simulation domain.

    Methods
    -------
    compute(x)
        Compute the PEI for a given input.
    add_repulsion_point(x)
        Add a new point to the set of repulsion points.
    expected_improvement(x)
        Calculate the expected improvement for a given input.
    repulsion(x)
        Compute the repulsion effect for a given input.

    Examples
    --------
    >>> domain = SimulatorDomain(...)
    >>> gp_model = AbstractGaussianProcess(...)
    >>> pei_calculator = PEICalculator(domain, gp_model)
    >>> pei_value = pei_calculator.compute(trial_point)

    Notes
    -----
    It computes the PEI for given inputs in a simulation domain, which features both expected
    improvement and a repulsion factor. Large values of pseudo-expected improvement indicate new
    inputs that reduce predictive uncertainty while not being too close to already-seen inputs.
    Optimising against PEI supports the search of experimental designs that balances exploration
    and exploitation of the input space.

    This class is tailored for Gaussian Process models and assumes properly initialised
    `AbstractGaussianProcess` and `SimulatorDomain` instances.
    """

    def __init__(self, domain: SimulatorDomain, gp: AbstractGaussianProcess):
        if not isinstance(domain, SimulatorDomain):
            raise TypeError(
                f"Expected 'domain' to be of type SimulatorDomain, but received {type(domain)} "
                "instead."
            )
        if not isinstance(gp, AbstractGaussianProcess):
            raise TypeError(
                f"Expected 'gp' to be of type AbstractGaussianProcess, but received {type(gp)} "
                "instead."
            )

        self._domain = domain
        self._gp = gp

        self._validate_training_data()

        self._max_targets = self._calculate_max_targets()
        self._repulsion_points = self._calculate_pseudopoints()

        self._standard_norm = norm(loc=0, scale=1)

    @property
    def gp(self) -> AbstractGaussianProcess:
        """(Read-only) The underlying AbstractGaussianProcess."""

        return self._gp

    @property
    def repulsion_points(self) -> tuple:
        """(Read-only) The current repulsion points."""

        return self._repulsion_points

    @property
    def domain(self) -> SimulatorDomain:
        """(Read-only) simulation domain."""

        return self._domain

    def _calculate_max_targets(self) -> Real:
        return max(self._gp.training_data, key=lambda datum: datum.output).output

    def _calculate_pseudopoints(self) -> tuple[Input]:
        training_data = [datum.input for datum in self._gp.training_data]
        return self._domain.calculate_pseudopoints(training_data)

    def _validate_training_data(self) -> None:
        if not self._gp.training_data:
            raise ValueError("'gp' training data is empty.")

        if not all(isinstance(datum, TrainingDatum) for datum in self._gp.training_data):
            raise TypeError(
                "All elements in 'gp' training data must be instances of TrainingDatum"
            )

    def _validate_input_type(
        self, x: Any, expected_types: tuple[Type, ...], method_name: str
    ) -> Input:
        if not isinstance(x, expected_types):
            raise TypeError(
                f"In method '{method_name}', expected 'x' to be of types {expected_types}, "
                f"but received {type(x)} instead."
            )

        if isinstance(x, np.ndarray):
            if x.ndim != 1:
                raise ValueError(
                    f"In method '{method_name}', the numpy ndarray 'x' must be one-dimensional."
                )
            return Input(*x)

        return x

    def compute(self, x: Union[Input, NDArray]) -> Real:
        """
        Compute the PseudoExpected Improvement (PEI) for a given input.

        This method calculates the PEI at a given point `x` by combining the expected improvement
        (EI) and the repulsion factor. The PEI is a metric used in Bayesian optimisation to balance
        exploration and exploitation, taking into account both the potential improvement over the
        current best target and the desire to explore less sampled regions of the domain.

        Parameters
        ----------
        x : Union[Input, NDArray]
            The input point for which to compute the PEI. This can be an instance of `Input` or
            a one-dimensional `numpy.ndarray`.

        Returns
        -------
        Real
            The computed PEI value for the given input. It is the product of the expected improvement
            and the repulsion factor.

        Raises
        ------
        TypeError
            If `x` is not an instance of `Input` or `numpy.ndarray`, or if `numpy.ndarray` is not one-dimensional.
        ValueError
            If `x` as `numpy.ndarray` is not one-dimensional.

        Examples
        --------
        >>> input_point = Input(2.0, 3.0)
        >>> pei = pei_calculator.compute(input_point)

        >>> array_input = np.array([2.0, 3.0])
        >>> pei = pei_calculator.compute(array_input)

        Notes
        -----
        The PEI is a product of two components: expected improvement, which quantifies the
        potential for improvement over the current maximum, and the repulsion factor, which
        discourages the selection of points near already sampled locations. This calculation
        assumes that the Gaussian Process model and other components of the system are properly
        initialised and configured.
        """

        return self.expected_improvement(x) * self.repulsion(x)

    def add_repulsion_point(self, x: Union[Input, NDArray]) -> None:
        """
        Add a new point to the set of repulsion points.

        This method updates the internal set of repulsion points used in the repulsion factor
        calculation. Repulsion points are used to influence the optimisation process, encouraging
        the exploration of new areas by reducing the likelihood of selecting points near these
        repulsion points.

        Parameters
        ----------
        x : Union[Input, NDArray]
            The point to be added to the repulsion points set. This can be an instance of `Input`
            or a one-dimensional `numpy.ndarray`. If `x` is a `numpy.ndarray`, it is converted
            to an `Input` object.

        Raises
        ------
        TypeError
            If `x` is not an instance of `Input` or `numpy.ndarray`, or if `numpy.ndarray` is not one-dimensional.
        ValueError
            If `x` as `numpy.ndarray` is not one-dimensional.

        Examples
        --------
        >>> new_repulsion_point = Input(4.0, 5.0)
        >>> pei_calculator.add_repulsion_point(new_repulsion_point)

        >>> array_repulsion_point = np.array([4.0, 5.0])
        >>> pei_calculator.add_repulsion_point(array_repulsion_point)
        """

        validated_x = self._validate_input_type(
            x, (Input, np.ndarray), "add_repulsion_point"
        )
        self._repulsion_points = self._repulsion_points + (validated_x,)

    def expected_improvement(self, x: Union[Input, NDArray]) -> Real:
        """
        Calculate the expected improvement (EI) for a given input.

        This method computes the EI of the given input point `x` using the Gaussian Process model.
        EI is a measure used in Bayesian optimisation and is particularly useful for guiding the
        selection of points in the domain where the objective function should be evaluated next.
        It is calculated based on the model's prediction at `x`, the current maximum target value,
        and the standard deviation of the prediction.

        Parameters
        ----------
        x : Union[Input, NDArray]
            The input point for which to calculate the expected improvement. This can be an instance
            of `Input` or a one-dimensional `numpy.ndarray`.

        Returns
        -------
        Real
            The expected improvement value for the given input. If the standard deviation of the
            prediction is zero (indicating no uncertainty), the EI is returned as 0.0.

        Raises
        ------
        TypeError
            If `x` is not an instance of `Input` or `numpy.ndarray`, or if `numpy.ndarray` is not one-dimensional.
        ValueError
            If `x` as `numpy.ndarray` is not one-dimensional.

        Examples
        --------
        >>> input_point = Input(1.0, 2.0)
        >>> ei = pei_calculator.expected_improvement(input_point)

        >>> array_input = np.array([1.0, 2.0])
        >>> ei = pei_calculator.expected_improvement(array_input)

        Notes
        -----
        The calculation of EI assumes that the Gaussian Process model (`self._gp`) has been
        trained with relevant data and that the current maximum target value (`self._max_targets`)
        has been computed from this data.
        """

        validated_x = self._validate_input_type(
            x, (Input, np.ndarray), "expected_improvement"
        )
        prediction = self._gp.predict(validated_x)

        if equal_within_tolerance(prediction.standard_deviation, 0):
            return 0.0

        u = (prediction.estimate - self._max_targets) / prediction.standard_deviation

        cdf_u = self._standard_norm.cdf(u)
        pdf_u = self._standard_norm.pdf(u)

        return (
            prediction.estimate - self._max_targets
        ) * cdf_u + prediction.standard_deviation * pdf_u

    def repulsion(self, x: Union[Input, NDArray]) -> Real:
        """
        Calculate the repulsion factor for a given input.

        This method assesses the repulsion effect of a given point `x` in relation to other points in the
        optimisation process. The repulsion factor is used to discourage the selection of points near
        already sampled locations, facilitating exploration of the domain. It is computed based on the
        covariance matrix of the input with respect to other points and the correlations derived from
        the Gaussian Process model.

        Parameters
        ----------
        x : Union[Input, NDArray]
            The input point for which to calculate the repulsion factor. This can be an instance
            of `Input` or a one-dimensional `numpy.ndarray`.

        Returns
        -------
        Real
            The repulsion factor for the given input. A higher value indicates a stronger repulsion
            effect, suggesting the point is near other sampled locations.

        Raises
        ------
        TypeError
            If `x` is not an instance of `Input` or `numpy.ndarray`, or if `numpy.ndarray` is not one-dimensional.
        ValueError
            If `x` as `numpy.ndarray` is not one-dimensional.

        Examples
        --------
        >>> input_point = Input(1.5, 2.5)
        >>> repulsion_factor = pei_calculator.repulsion(input_point)

        >>> array_input = np.array([1.5, 2.5])
        >>> repulsion_factor = pei_calculator.repulsion(array_input)

        Notes
        -----
        The repulsion factor calculation relies on the Gaussian Process model (`self._gp`) and the
        set of other repulsion points (`self._repulsion_points`). It is important that the
        Gaussian Process model is properly trained and that relevant repulsion points are added
        to the system for accurate calculations.
        """

        validated_x = self._validate_input_type(x, (Input, np.ndarray), "repulsion")

        covariance_matrix = self._gp.covariance_matrix([validated_x])
        correlations = (
            np.array(covariance_matrix) / self._gp.fit_hyperparameters.process_var
        )
        inputs_term = np.product(1 - correlations, axis=0)[0]

        other_repulsion_pts_term = np.product(
            1 - np.array(self._gp.correlation([validated_x], self._repulsion_points)),
            axis=1,
        )[0]

        return inputs_term * other_repulsion_pts_term


def compute_single_level_loo_samples(
    gp: AbstractGaussianProcess,
    domain: SimulatorDomain,
    batch_size: int = 1,
    loo_errors_gp: Optional[AbstractGaussianProcess] = None,
) -> tuple[Input]:
    gp_e = compute_loo_errors_gp(gp, domain)
    pei = PEICalculator(domain, gp_e)

    # TODO: correct the implementation to iteratively use updated PEI function
    return (maximise(lambda x: pei.compute(x), domain),) * batch_size
