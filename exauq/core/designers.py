import copy
import itertools
import math
from collections.abc import Sequence
from numbers import Real
from typing import Any, Optional, Type, Union

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from exauq.core.modelling import (
    AbstractGaussianProcess,
    GaussianProcessPrediction,
    Input,
    MultiLevel,
    MultiLevelGaussianProcess,
    OptionalFloatPairs,
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


def compute_loo_errors_gp(
    gp: AbstractGaussianProcess,
    domain: SimulatorDomain,
    loo_errors_gp: Optional[AbstractGaussianProcess] = None,
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
    loo_errors_gp : Optional[AbstractGaussianProcess], optional
        (Default: None) Another Gaussian process that is trained on the LOO errors to
        create the output to this function. If ``None`` then a deep copy of `gp` will
        be used instead.

    Returns
    -------
    AbstractGaussianProcess
        A Gaussian process that is trained on the normalised expected square LOO errors
        of `gp`. If `loo_errors_gp` was supplied then (a reference to) this object will be
        returned (except now it has been fit to the LOO errors).

    Raises
    ------
    ValueError
        If any of the training inputs in `gp` do not belong to the simulator domain `domain`.

    Notes
    -----
    The lower bounds on the correlation length scale parameters are obtained by
    multiplying the lengths of the domain's dimensions by ``sqrt(-0.5 / log(10 ** (-8)))``.
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

    if not (loo_errors_gp is None or isinstance(loo_errors_gp, AbstractGaussianProcess)):
        raise TypeError(
            "Expected 'loo_errors_gp' to be None or of type AbstractGaussianProcess, but "
            f"received {type(loo_errors_gp)} instead."
        )

    error_training_data = []
    loo_gp = copy.deepcopy(gp)
    for leave_out_idx, datum in enumerate(gp.training_data):
        # Fit LOO GP, storing into loo_gp
        _ = compute_loo_gp(gp, leave_out_idx, loo_gp=loo_gp)

        # Add training input and nes error
        loo_prediction = loo_gp.predict(datum.input)
        nes_loo_error = loo_prediction.nes_error(datum.output)
        error_training_data.append(TrainingDatum(datum.input, nes_loo_error))

    gp_e = loo_errors_gp if loo_errors_gp is not None else copy.deepcopy(gp)
    bounds = _compute_loo_error_bounds(domain)
    gp_e.fit(error_training_data, hyperparameter_bounds=bounds)
    return gp_e


def _compute_loo_error_bounds(domain: SimulatorDomain) -> Sequence[OptionalFloatPairs]:
    """Compute bounds on correlation length scale parameters to use when fitting a
    Gaussian process to leave-one-out errors.

    This is as specified in Mohammadi, H. et al. (2022) "Cross-Validation-based Adaptive
    Sampling for Gaussian process models". DOI: https://doi.org/10.1137/21M1404260
    """

    # Note: the following is a simplification of sqrt(-0.5 / log(10 ** (-8))) from paper
    bound_scale = 0.25 / math.sqrt(math.log(10))
    return [(bound_scale * (bnd[1] - bnd[0]), None) for bnd in domain.bounds] + [
        (None, None)
    ]


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

    if len(gp.training_data) < 2:
        raise ValueError(
            "Cannot compute leave one out error with 'gp' because it has not been trained "
            "on at least 2 data points."
        )

    for dat1, dat2 in itertools.combinations(gp.training_data, 2):
        if dat1.input == dat2.input:
            raise ValueError(
                "Cannot compute leave one out error with 'gp' because simulator input "
                f"{dat1.input} is repeated in the training data."
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

    Implements the PEI detailed in Mohammadi et. al. (2022)[1]_. The functionality in this
    class applies to Gaussian Process models that have been trained on data with inputs
    lying in the supplied simulator domain.

    Parameters
    ----------
    domain : SimulatorDomain
        The domain of a simulation.
    gp : AbstractGaussianProcess
        A Gaussian process model, which is trained on data where the simulator inputs are
        in `domain`.

    Attributes
    ----------
    gp : AbstractGaussianProcess
        (Read-only) The Gaussian process used in calculations.
    repulsion_points : tuple[Input]
        (Read-only) The current set of repulsion points used in calculations.
    domain : SimulatorDomain
        (Read-only) The simulator domain used in calculations.

    Examples
    --------
    >>> domain = SimulatorDomain(...)
    >>> gp_model = AbstractGaussianProcess(...)
    >>> pei_calculator = PEICalculator(domain, gp_model)
    >>> pei_value = pei_calculator.compute(trial_point)

    Notes
    -----
    This class computes the PEI for given inputs in a simulation domain, which features both expected
    improvement and a repulsion factor. Large values of pseudo-expected improvement indicate new
    inputs that reduce predictive uncertainty while not being too close to already-seen inputs.
    Optimising against PEI supports the search of experimental designs that balances exploration
    and exploitation of the input space.

    References
    ----------
    .. [1] Mohammadi, H. et al. (2022) "Cross-Validation-based Adaptive Sampling for
        Gaussian process models". DOI: https://doi.org/10.1137/21M1404260
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
        self._other_repulsion_points = self._calculate_pseudopoints()

        self._standard_norm = norm(loc=0, scale=1)

    @property
    def gp(self) -> AbstractGaussianProcess:
        """(Read-only) The Gaussian process used in calculations."""

        return self._gp

    @property
    def repulsion_points(self) -> tuple[Input]:
        """(Read-only) The current set of repulsion points used in calculations."""
        training_points = tuple(datum.input for datum in self._gp.training_data)
        return self._other_repulsion_points + training_points

    @property
    def domain(self) -> SimulatorDomain:
        """(Read-only) The simulator domain used in calculations."""

        return self._domain

    def _calculate_max_targets(self) -> Real:
        return max(self._gp.training_data, key=lambda datum: datum.output).output

    def _calculate_pseudopoints(self) -> tuple[Input]:
        training_data = [datum.input for datum in self._gp.training_data]
        return self._domain.calculate_pseudopoints(training_data)

    def _validate_training_data(self) -> None:
        if not self._gp.training_data:
            raise ValueError("Expected 'gp' to have nonempty training data.")

        if not all(isinstance(datum, TrainingDatum) for datum in self._gp.training_data):
            raise TypeError(
                "All elements in 'gp' training data must be instances of TrainingDatum"
            )

    def _validate_input_type(
        self, x: Any, expected_types: tuple[Type, ...], method_name: str
    ) -> Input:
        if not isinstance(x, expected_types):
            raise TypeError(
                f"In method '{method_name}', expected 'x' to be one of the types {expected_types}, "
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

        Examples
        --------
        >>> input_point = Input(2.0, 3.0)
        >>> pei = pei_calculator.compute(input_point)

        >>> array_input = np.array([2.0, 3.0])
        >>> pei = pei_calculator.compute(array_input)

        Notes
        -----
        This method calculates the PEI at a given point `x` by combining the expected improvement
        (EI) and the repulsion factor. The PEI is a metric used in Bayesian optimisation to balance
        exploration and exploitation, taking into account both the potential improvement over the
        current best target and the desire to explore less sampled regions of the domain.
        """

        return self.expected_improvement(x) * self.repulsion(x)

    def add_repulsion_point(self, x: Union[Input, NDArray]) -> None:
        """
        Add a new point to the set of repulsion points.

        This method updates the internal set of repulsion points used in the repulsion factor
        calculation. Simulator inputs very positively correlated with repulsion points result in low
        repulsion values, whereas inputs very negatively correlated with repulsion points result
        in high repulsion values.

        Parameters
        ----------
        x : Union[Input, NDArray]
            The point to be added to the repulsion points set. This can be an instance of `Input`
            or a one-dimensional `numpy.ndarray`. If `x` is a `numpy.ndarray`, it is converted
            to an `Input` object.

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
        self._other_repulsion_points = self._other_repulsion_points + (validated_x,)

    def expected_improvement(self, x: Union[Input, NDArray]) -> Real:
        """
        Calculate the expected improvement (EI) for a given input.

        If the standard deviation of the prediction is within the default
        tolerance ``exauq.core.numerics.FLOAT_TOLERANCE`` of 0 then the EI returned is 0.0.

        Parameters
        ----------
        x : Union[Input, NDArray]
            The input point for which to calculate the expected improvement. This can be an instance
            of `Input` or a one-dimensional `numpy.ndarray`.

        Returns
        -------
        Real
            The expected improvement value for the given input.

        Examples
        --------
        >>> input_point = Input(1.0, 2.0)
        >>> ei = pei_calculator.expected_improvement(input_point)

        >>> array_input = np.array([1.0, 2.0])
        >>> ei = pei_calculator.expected_improvement(array_input)

        Notes
        -----
        This method computes the EI of the given input point `x` using the Gaussian Process model.
        EI is a measure used in Bayesian optimisation and is particularly useful for guiding the
        selection of points in the domain where the objective function should be evaluated next.
        It is calculated based on the model's prediction at `x`, the current maximum target value,
        and the standard deviation of the prediction.
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
        Calculate the repulsion factor for a given simulator input.

        This method assesses the repulsion effect of a given point `x` in relation to other,
        stored repulsion points. It is calculated as the product of terms
        ``1 - correlation(x, rp)``, where ``rp`` is a repulsion point and the correlation is
        computed with the Gaussian process supplied at this object's initialisation. The
        repulsion factor can be used to discourage the selection of points near already
        sampled locations, facilitating exploration of the input space.

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

        Examples
        --------
        >>> input_point = Input(1.5, 2.5)
        >>> repulsion_factor = pei_calculator.repulsion(input_point)

        >>> array_input = np.array([1.5, 2.5])
        >>> repulsion_factor = pei_calculator.repulsion(array_input)
        """

        validated_x = self._validate_input_type(x, (Input, np.ndarray), "repulsion")

        covariance_matrix = self._gp.covariance_matrix([validated_x])
        correlations = (
            np.array(covariance_matrix) / self._gp.fit_hyperparameters.process_var
        )
        inputs_term = np.prod(1 - correlations, axis=1)

        other_repulsion_pts_term = np.prod(
            1
            - np.array(self._gp.correlation([validated_x], self._other_repulsion_points)),
            axis=1,
        )

        return float(inputs_term * other_repulsion_pts_term)


def compute_single_level_loo_samples(
    gp: AbstractGaussianProcess,
    domain: SimulatorDomain,
    batch_size: int = 1,
    loo_errors_gp: Optional[AbstractGaussianProcess] = None,
    seed: Optional[int] = None,
) -> tuple[Input]:
    """Compute a new batch of design points adaptively for a single-level Gaussian process.

    Implements the cross-validation-based adaptive sampling for emulators, as described in
    Mohammadi et. al. (2022)[1]_. This involves computing a Gaussian process (GP) trained
    on normalised expected squared errors arising from a leave-one-out (LOO)
    cross-validation, then finding design points that maximise the pseudo-expected
    improvement of this LOO errors GP.

    By default, a deep copy of the main GP supplied (`gp`) is trained on the leave-one-out
    errors. Alternatively, another ``AbstractGaussianProcess`` can be supplied that will
    be trained on the leave-one-out errors (and so modified in-place), allowing for the
    use of different Gaussian process settings (e.g. a different kernel function).

    If `seed` is provided, then this will be used when maximising the pseudo-expected
    improvement of the LOO errors GP. Providing a seed does not necessarily mean
    calculation of the output design points is deterministic, as this also depends on
    computation of the LOO errors GP being deterministic.

    Parameters
    ----------
    gp : AbstractGaussianProcess
        A Gaussian process to compute new design points for.
    domain : SimulatorDomain
        The domain of a simulator that the Gaussian process `gp` emulates. The data on
        which `gp` is trained are expected to have simulator inputs only from this domain.
    batch_size : int, optional
        (Default: 1) The number of new design points to compute. Should be a positive
        integer.
    loo_errors_gp : AbstractGaussianProcess, optional
        (Default: None) Another Gaussian process that is trained on the LOO errors as part
        of the adaptive sampling method. If ``None`` then a deep copy of `gp` will be used
        instead.
    seed : int, optional
        (Default: None) A random number seed to use when maximising pseudo-expected
        improvement. If ``None`` then the maximisation won't be seeded.

    Returns
    -------
    tuple[Input]
        The batch of new design points.

    Raises
    ------
    ValueError
        If any of the training inputs in `gp` do not belong to the simulator domain `domain`.

    See Also
    --------
    compute_loo_errors_gp :
        Computation of the leave-one-out errors Gaussian process.
    PEICalculator :
        Pseudo-expected improvement calculation.
    modelling.GaussianProcessPrediction.nes_error :
        Normalised expected squared error for a prediction from a Gaussian process.
    utilities.optimisation.maximise :
        Global maximisation over a simulator domain, used on pseudo-expected improvement
        for the LOO errors GP.

    References
    ----------
    .. [1] Mohammadi, H. et al. (2022) "Cross-Validation-based Adaptive Sampling for
        Gaussian process models". DOI: https://doi.org/10.1137/21M1404260
    """
    if not isinstance(batch_size, int):
        raise TypeError(
            f"Expected 'batch_size' to be an integer, but received {type(batch_size)} instead."
        )

    if batch_size < 1:
        raise ValueError(
            f"Expected batch size to be a positive integer, but received {batch_size} instead."
        )

    try:
        gp_e = compute_loo_errors_gp(gp, domain, loo_errors_gp=loo_errors_gp)
    except (TypeError, ValueError) as e:
        raise e from None

    pei = PEICalculator(domain, gp_e)

    design_points = []
    for _ in range(batch_size):
        new_design_point, _ = maximise(lambda x: pei.compute(x), domain, seed=seed)
        design_points.append(new_design_point)
        pei.add_repulsion_point(new_design_point)

    return tuple(design_points)


def compute_multi_level_loo_prediction(
    mlgp: MultiLevelGaussianProcess,
    level: int,
    leave_out_idx: int,
    loo_gp: Optional[AbstractGaussianProcess] = None,
) -> GaussianProcessPrediction:
    """Make a prediction from a multi-level leave-one-out (LOO) Gaussian process (GP) at
    the left out point.

    The multi-level LOO prediction at the left-out simulator input is a sum of
    predictions made for each level in the given multi-level GP. The contribution at the
    level containing the left out training datum (defined by `level`) is the prediction
    made by the LOO GP at the given level in `mlgp` (see ``compute_loo_prediction``). The
    contributions at the other levels are based on predictions made by the GPs in `mlgp`
    at these levels under the assumption of a zero prior mean.

    The formula for calculating the leave-one-out prediction assumes that none of the
    levels in the multi-level GP share common training simulator inputs; a ValueError
    will be raised if this is not the case.

    Parameters
    ----------
    mlgp : MultiLevelGaussianProcess
        A multi-level Gaussian process to form the basis for the multi-level LOO GP.
    level : int
        The level containing the datum to leave out.
    leave_out_idx : int
        The index for the training datum of `mlgp` to leave out, at the level `level`.
    loo_gp : AbstractGaussianProcess, optional
        (Default: None) A Gaussian process that is trained on the LOO data and then
        used to make the prediction for `level` at the left-out simulator input. If
        ``None`` then a deep copy of the GP at level `level` in `mlgp` will be used
        instead.

    Returns
    -------
    GaussianProcessPrediction
        The prediction at the left out training input, based on the multi-level LOO GP
        described above.

    Raises
    ------
    ValueError
        If there is a shared training simulator input across multiple levels in `mlgp`.
    """

    n_training_data = len(mlgp.training_data[level])
    if not 0 <= leave_out_idx < n_training_data:
        raise ValueError(
            "'leave_out_idx' should define a zero-based index for the training data "
            f"of length {n_training_data} at level {level}, but received out of range "
            f"index {leave_out_idx}."
        )

    if repetition := _find_input_repetition_across_levels(mlgp.training_data):
        repeated_input, level1, level2 = repetition
        raise ValueError(
            "Training inputs across all levels must be distinct, but found common "
            f"input {repeated_input} at levels {level1}, {level2}."
        )

    terms = MultiLevel({level: None for level in mlgp.levels})

    # Get mean and variance contributions at supplied level
    terms[level] = compute_loo_prediction(mlgp[level], leave_out_idx, loo_gp=loo_gp)

    # Get mean and variance contributions at other levels
    loo_input = mlgp.training_data[level][leave_out_idx].input
    terms.update(
        {
            j: compute_zero_mean_prediction(mlgp[j], loo_input)
            for j in mlgp.levels
            if not j == level
        }
    )

    # Aggregate predictions across levels
    mean = sum(mlgp.coefficients[level] * terms[level].estimate for level in terms.levels)
    variance = sum(
        (mlgp.coefficients[level] ** 2) * terms[level].variance for level in terms.levels
    )

    return GaussianProcessPrediction(mean, variance)


def _find_input_repetition_across_levels(
    training_data: MultiLevel[Sequence[TrainingDatum]],
) -> Optional[tuple[Input, int, int]]:
    """Find a training input that features in multiple levels, if such an input exists.

    If a repeated input is found, a triple ``(repeated_input, level1, level2)`` is
    returned, where ``repeated_input`` is the repeated input and ``level1``, ``level2``
    are the levels where repetition occurs. If no repeated inputs are found, return
    ``None``.

    Note that repetition is determined by testing for equality between input objects and
    is only applied to inputs on different levels.
    """
    training_inputs = tuple(
        itertools.starmap(
            lambda level, data: ((level, datum.input) for datum in data),
            training_data.items(),
        )
    )

    # If there is only a single level then we don't check for repetitions
    # of inputs on the same level, so return.
    if len(training_inputs) < 2:
        return None

    for levels_and_inputs in itertools.product(*training_inputs):
        for (level1, x1), (level2, x2) in itertools.combinations(levels_and_inputs, 2):
            if x1 == x2:
                return x1, level1, level2

    return None


def compute_loo_prediction(
    gp, leave_out_idx, loo_gp: Optional[AbstractGaussianProcess] = None
) -> GaussianProcessPrediction:
    """Make a prediction from a leave-one-out (LOO) Gaussian process (GP) at the left out
    point.

    The LOO Gaussian process (GP) is obtained by training it on all training data from the
    supplied GP except for one datum (the 'left out' datum). It is trained using the
    fitted hyperparameters *from the supplied Gaussian process `gp`*.

    By default, the LOO GP used will be a deep copy of `gp` trained on the leave-one-out
    data. Alternatively, another ``AbstractGaussianProcess`` can be supplied that will be
    trained on the leave-one-out data. This can be more efficient when repeated
    calculation of a LOO GP is required.

    Parameters
    ----------
    gp : _type_
        A Gaussian process to form the basis for the LOO GP.
    leave_out_idx : _type_
        The index for the training datum of `gp` to leave out. This should be an index
        of the sequence returned by the ``gp.training_data`` property.
    loo_gp : AbstractGaussianProcess, optional
        (Default: None) Another Gaussian process that is trained on the LOO data and then
        used to make the prediction at the left-out simulator input. If ``None`` then a
        deep copy of `gp` will be used instead.

    Returns
    -------
    GaussianProcessPrediction
        The prediction of the LOO Gaussian process at the left out simulator input.
    """
    # Get left-out training data
    loo_input = gp.training_data[leave_out_idx].input
    loo_output = gp.training_data[leave_out_idx].output

    loo_prediction = compute_loo_gp(gp, leave_out_idx, loo_gp=loo_gp).predict(loo_input)

    return GaussianProcessPrediction(
        loo_prediction.estimate - loo_output, loo_prediction.variance
    )


def compute_zero_mean_prediction(
    gp: AbstractGaussianProcess, x: Input
) -> GaussianProcessPrediction:
    """Make a prediction at an input based on a Gaussian process but with zero prior mean.

    Parameters
    ----------
    gp : AbstractGaussianProcess
        A Gaussian process.
    x : Input
        A simulator input to make the prediction at.

    Returns
    -------
    GaussianProcessPrediction
        The prediction made at `x` by a Gaussian process having the same covariance as
        `gp` but zero prior mean.

    Raises
    ------
    ValueError
        If `gp` hasn't been trained on any data.
    """
    training_inputs = [datum.input for datum in gp.training_data]
    training_outputs = np.array([[datum.output] for datum in gp.training_data])

    try:
        mean = float(
            gp.covariance_matrix([x])
            @ np.linalg.inv(gp.covariance_matrix(training_inputs))
            @ training_outputs
        )
    except ValueError:
        if not training_inputs:
            raise ValueError(
                "Cannot calculate zero-mean prediction: 'gp' hasn't been trained on any data."
            )
    except np.linalg.LinAlgError as e:
        raise ValueError(f"Cannot calculate zero-mean prediction: {e}")

    variance = gp.predict(x).variance
    return GaussianProcessPrediction(mean, variance)


def compute_multi_level_loo_error_data(
    mlgp: MultiLevelGaussianProcess,
) -> MultiLevel[tuple[TrainingDatum]]:
    """Calculate multi-level leave-one-out (LOO) errors.

    This involves computing normalised expected squared errors for GPs based on a
    leave-one-out (LOO) cross-validation across all the levels. For each simulator input
    in the training data of `mlgp`, the normalised expected squared error of the
    prediction of an intermediary LOO multi-level GP at the input is calculated. The
    intermediary LOO multi-level GP involves computing a single-level LOO GP at the level
    of the left out input (see `compute_loo_errors_gp`). A copy of the GP at the
    appropriate level in `mlgp` is used for this intermediary LOO GP and the corresponding
    hyperparameters found in `mlgp` are used when fitting to the data.

    Parameters
    ----------
    mlgp : MultiLevelGaussianProcess
        A multi-level GP to calculate the errors for.

    Returns
    -------
    MultiLevel[tuple[TrainingDatum]]
        Multi-level data consisting of the simulator inputs from the training data for
        `mlgp` and the errors arising from the corresponding leave-one-out calculations.

    Raises
    ------
    ValueError
        If the GP at some level within `mlgp` has not been trained on more than one datum.

    See Also
    --------
    compute_multi_level_loo_prediction :
        Calculate the prediction of a LOO multi-level GP at a left-out training input.

    """

    training_data_counts = mlgp.training_data.map(lambda level, data: len(data))
    if levels_not_enough_data := {
        str(level) for level, count in training_data_counts.items() if count < 2
    }:
        bad_levels = ", ".join(sorted(levels_not_enough_data))
        raise ValueError(
            f"Could not perform leave-one-out calculation: levels {bad_levels} not "
            "trained on at least two training data."
        )

    error_training_data = MultiLevel([[] for _ in mlgp.levels])
    for level, gp in mlgp.items():
        # Create a copy for the leave-one-out computations, for efficiency.
        loo_gp = copy.deepcopy(gp)

        for leave_out_index, datum in enumerate(gp.training_data):
            loo_prediction = compute_multi_level_loo_prediction(
                mlgp, level, leave_out_index, loo_gp=loo_gp
            )
            error_training_data[level].append(
                TrainingDatum(datum.input, loo_prediction.nes_error(0))
            )

    return error_training_data.map(lambda level, data: tuple(data))


def compute_multi_level_loo_errors_gp(
    mlgp: MultiLevelGaussianProcess,
    domain: SimulatorDomain,
    output_mlgp: Optional[MultiLevelGaussianProcess] = None,
) -> MultiLevelGaussianProcess:
    """Calculate the multi-level Gaussian process (GP) trained on normalised expected
    squared leave-one-out (LOO) errors.

    The returned multi-level GP is obtained by training it on training data calculated
    with ``compute_multi_level_loo_error_data``. This involves computing normalised
    expected squared errors for GPs based on a leave-one-out (LOO) cross-validation across
    all the levels. The resulting errors, together with the corresponding left out
    simulator inputs, form the training data for the output GP. The
    output GP is trained with a lower bound on the correlation length scale parameters for
    each level (see the Notes section).

    By default, the returned multi-level GP will be a deep copy of `mlgp` trained on the
    error data. Alternatively, another multi-level GP can be supplied that will be trained
    on the error data and returned (thus it will be modified in-place as well as
    returned).

    Parameters
    ----------
    mlgp : MultiLevelGaussianProcess
        A multi-level GP to calculate the normalised expected squared LOO errors for.
    domain : SimulatorDomain
        The domain of a simulator that the multi-level Gaussian process `mlgp` emulates.
        The data on which each level of `mlgp` is trained are expected to have simulator
        inputs only from this domain.
    output_mlgp : MultiLevelGaussianProcess, optional
        (Default: None) Another multi-level GP that is trained on the LOO errors to
        create the output to this function. If ``None`` then a deep copy of `mlgp` will
        be used instead.

    Returns
    -------
    MultiLevelGaussianProcess
        A multi-level GP that is trained on the normalised expected square LOO errors
        arising from `mlgp`. If `output_mlgp` was supplied then (a reference to) this
        object will be returned (except now it has been fit to the LOO errors data).

    Raises
    ------
    ValueError
        If the set of levels for `output_mlgp` is not equal to those in `mlgp` (when
        `output_mlgp` is not ``None``).

    See Also
    --------
    compute_multi_level_loo_error_data :
        Calculation of the leave-one-out error data used to train this function's output.

    Notes
    -----
    The lower bounds on the correlation length scale parameters are obtained by
    multiplying the lengths of the domain's dimensions by ``sqrt(-0.5 / log(10 ** (-8)))``.
    Note that the same lower bounds are used for each level.
    """

    if output_mlgp is not None and not mlgp.levels == output_mlgp.levels:
        raise ValueError(
            f"Expected the levels {output_mlgp.levels} of 'output_mlgp' to match the levels "
            f"{mlgp.levels} from 'mlgp'."
        )

    # Create LOO errors for each level
    error_training_data = compute_multi_level_loo_error_data(mlgp)

    # Train GP on the LOO errors
    ml_errors_gp = output_mlgp if output_mlgp is not None else copy.deepcopy(mlgp)
    ml_errors_gp.fit(
        error_training_data, hyperparameter_bounds=_compute_loo_error_bounds(domain)
    )
    return ml_errors_gp


def compute_multi_level_loo_samples(
    mlgp: MultiLevelGaussianProcess,
    domain: SimulatorDomain,
    costs: MultiLevel[Real],
    batch_size: int = 1,
) -> tuple[int, tuple[Input, ...]]:
    """Compute a batch of design points adaptively for a multi-level Gaussian process (GP).

    Implements the cross-validation-based adaptive sampling for multi-level Gaussian
    process models, as described in Kimpton et. al. (2024)[1]_. This involves computing a
    multi-level GP that is trained on normalised expected squared errors arising from a
    multi-level leave-one-out (LOO) cross-validation. The design points returned are those
    that maximise weighted pseudo-expected improvements (PEIs) of this multi-level LOO
    errors GP across levels, where the PEIs are weighted according to the costs of
    computing the design points on simulators at the levels.

    The `costs` should represent the costs of running a each level's simulator on a single
    input.

    The adaptive sampling method assumes that none of the levels in the multi-level GP
    share common training simulator inputs; a ValueError will be raised if this is not the
    case.

    Parameters
    ----------
    mlgp : MultiLevelGaussianProcess
        The multi-level GP to create the design points for.
    domain : SimulatorDomain
        The domain of a simulator that the multi-level Gaussian process `mlgp` emulates.
        The data on which each level of `mlgp` is trained are expected to have simulator
        inputs only from this domain.
    costs : MultiLevel[Real]
        The costs of running a simulation at each of the levels.
    batch_size : int, optional
        (Default: 1) The number of new design points to compute. Should be a positive
        integer.

    Returns
    -------
    tuple[int, tuple[Input, ...]]
        A pair ``(level, data)`` where ``data`` is the batch of design points and
        ``level`` is the level of simulation at which the design point should be run.

    Raises
    ------
    ValueError
        If any of the training inputs in `mlgp` do not belong to the simulator domain
        `domain`.
    ValueError
        If any of the levels defined in `mlgp` does not have an associated cost.
    ValueError
        If there is a shared training simulator input across multiple levels in `mlgp`.

    See Also
    --------
    compute_multi_level_loo_errors_gp :
        Computation of the multi-level leave-one-out errors GP.
    PEICalculator :
        Pseudo-expected improvement calculation.
    modelling.GaussianProcessPrediction.nes_error :
        Normalised expected squared error for a prediction from a Gaussian process.
    utilities.optimisation.maximise :
        Global maximisation over a simulator domain, used on pseudo-expected improvement
        for the multi-level LOO errors GP.

    References
    ----------
    .. [1] Kimpton, L. M. et al. (2023) "Cross-Validation Based Adaptive Sampling for
        Multi-Level Gaussian Process Models". arXiv: https://arxiv.org/abs/2307.09095
    """

    if not isinstance(mlgp, MultiLevelGaussianProcess):
        raise TypeError(
            f"Expected 'mlgp' to be of type {MultiLevelGaussianProcess}, but "
            f"received {type(mlgp)} instead."
        )
    if not isinstance(domain, SimulatorDomain):
        raise TypeError(
            f"Expected 'domain' to be of type {SimulatorDomain}, but received "
            f"{type(domain)} instead."
        )

    if not all(
        datum.input in domain
        for level in mlgp.levels
        for datum in mlgp.training_data[level]
    ):
        raise ValueError(
            "Expected all training inputs in 'mlgp' to belong to the domain 'domain', but "
            "this is not the case."
        )

    if missing_levels := sorted(set(mlgp.levels) - set(costs.levels)):
        raise ValueError(
            f"Level {missing_levels[0]} from 'mlgp' does not have associated level "
            "from 'costs'."
        )

    if not isinstance(batch_size, int):
        raise TypeError(
            f"Expected 'batch_size' to be an integer, but received {type(batch_size)} instead."
        )

    if batch_size < 1:
        raise ValueError(
            f"Expected batch size to be a positive integer, but received {batch_size} instead."
        )

    # Create LOO errors GP for each level
    ml_errors_gp = compute_multi_level_loo_errors_gp(mlgp, domain, output_mlgp=None)

    # Get the PEI calculator for each level
    ml_pei = ml_errors_gp.map(lambda _, gp: PEICalculator(domain, gp))

    # TODO: change costs to apply as-is rather than deltas.
    # Find PEI argmax, with (weighted) PEI value, for each level
    delta_costs = costs.map(lambda level, _: _compute_delta_cost(costs, level))
    maximal_pei_values = ml_pei.map(
        lambda level, pei: maximise(lambda x: pei.compute(x) / delta_costs[level], domain)
    )

    # Create batch at maximising level by iteratively adding new design points as
    # repulsion points and re-maximising PEI
    level, (x, _) = max(maximal_pei_values.items(), key=lambda item: item[1][1])
    design_points = [x]
    if batch_size > 1:
        pei = ml_pei[level]
        for i in range(batch_size - 1):
            pei.add_repulsion_point(design_points[i])
            new_design_pt, _ = maximise(lambda x: pei.compute(x), domain)
            design_points.append(new_design_pt)

    return level, tuple(design_points)


def _compute_delta_cost(costs: MultiLevel[Real], level: int) -> Real:
    """Compute the cost of computing a successive difference of simulations at a level."""

    if level == 1:
        return costs[1]
    else:
        return costs[level - 1] + costs[level]
