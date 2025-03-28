"""
Create the experimental design using either a simple one-shot Latin hypercube or
through the LOO sampling methods for both single and multi-level GPs. Repulsion
points can also be added and their effect calculated through the pseudo expected
improvement (PEI) class.  


Sampling Methods
------------------------------------------------------------------------------------------------
[`compute_single_level_loo_samples`][exauq.core.designers.compute_single_level_loo_samples]
Single level leave-one-out design points.

[`compute_multi_level_loo_samples`][exauq.core.designers.compute_multi_level_loo_samples]
Multi level leave-one-out design points.

[`oneshot_lhs`][exauq.core.designers.oneshot_lhs]
Latin hypercube sampling.


[PEI Calculator][exauq.core.designers.PEICalculator]
------------------------------------------------------------------------------------------------
[`repulsion_points`][exauq.core.designers.PEICalculator.repulsion_points]
Current set of repulsion points.

[`add_repulsion_points`][exauq.core.designers.PEICalculator.add_repulsion_points]
Add simulator points to repulsion points.

[`compute`][exauq.core.designers.PEICalculator.compute]
Compute pseudo expected improvement for a given input.

[`expected_improvement`][exauq.core.designers.PEICalculator.expected_improvement]
Calculate expected improvement for a given input.

[`repulsion`][exauq.core.designers.PEICalculator.repulsion]
Calculate repulsion factor for a given input.


Computing Leave-One-Out
------------------------------------------------------------------------------------------------
[`compute_loo_gp`][exauq.core.designers.compute_single_level_loo_samples]
Calculate a leave-one-out GP.

[`compute_loo_errors_gp`][exauq.core.designers.compute_loo_errors_gp]
Calculate a GP trained on leave-one-out errors.

[`compute_loo_prediction`][exauq.core.designers.compute_loo_prediction]
Make a prediction from GP minus leave-one-out point.

[`compute_multi_level_loo_error_data`][exauq.core.designers.compute_multi_level_loo_error_data]
Calculate multi-level leave-one-out errors.

[`compute_multi_level_loo_errors_gp`][exauq.core.designers.compute_multi_level_loo_errors_gp]
Calculate a GP trained on multi-level leave-one-out errors.

[`compute_multi_level_loo_prediction`][exauq.core.designers.compute_multi_level_loo_prediction]
Make a prediction for a Multi-level GP minus leave-one-out point.


"""

import copy
import itertools
import math
from collections import defaultdict
from collections.abc import Collection, Sequence
from numbers import Real
from typing import Any, Optional, Union
from warnings import warn

import numpy as np
from scipy.stats import norm
from scipy.stats.qmc import LatinHypercube

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
from exauq.utilities.optimisation import generate_seeds, maximise
from exauq.utilities.validation import check_int


def _check_collection_of_inputs(coll: Any, name: str):
    """Check whether the given `coll` is a collection of ``Input`` objects, raising a
    TypeError if not. `name` is used to refer to the collection in error messages."""

    if not isinstance(coll, Collection):
        raise TypeError(
            f"Expected '{name}' to be of type collection of {Input}s,"
            f"but received {type(coll)} instead."
        )
    elif any(not isinstance(x, Input) for x in coll):
        raise TypeError(
            f"Expected '{name}' to be of type collection of {Input}s,"
            f"but one or more elements were of an unexpected type."
        )
    else:
        pass


def _find_input_outside_domain(
    inputs: Collection[Input], domain: SimulatorDomain
) -> Optional[Input]:
    """Find an input not contained in `domain` from the given collection `inputs`, or
    return None if no such input exists."""

    for x in inputs:
        if x not in domain:
            return x
        else:
            continue

    return None


def oneshot_lhs(
    domain: SimulatorDomain, batch_size: int, seed: Optional[int] = None
) -> tuple[Input, ...]:
    """
    Create a "one-shot" design for a simulator using the Latin hypercube method.

    The Latin hypercube sample generates points in the unit square to spatially fill the
    domain as best as possible. It is then rescaled to match the design of the simulator.
    The algorithm is implemented from the Scipy package using the provided domain and
    number of design points chosen (see notes for further details).

    Parameters
    ----------
    domain :
        The domain of a simulator, defining the bounded input space over which the Latin
        hypercube will be generated.

    batch_size :
        The number of design points to create within the domain.

    seed :
        A number to seed the random number generator used in the
        underlying optimisation. If ``None`` then no seeding will be used.

    Returns
    -------
    tuple[Input, ...]
        The inputs for the domain generated by the Latin hypercube and scaled to match
        the design of the simulator, returned as a tuple of inputs.

    Notes
    -----
    The Scipy documentation for the Latin hypercube:
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.LatinHypercube.html>

    """

    if not isinstance(domain, SimulatorDomain):
        raise TypeError(
            f"Expected 'domain' to be of type SimulatorDomain, but received {type(domain)} instead."
        )

    check_int(
        batch_size,
        TypeError(
            f"Expected 'batch_size' to be of type int, but received {type(batch_size)} instead."
        ),
    )
    if batch_size <= 0:
        raise ValueError(
            f"Expected 'batch_size' to be a non-negative integer >0 but is equal to {batch_size}."
        )

    # Use the dimension of the domain in defining the Latin hypercube sampler.
    # Seed used to make the sampling repeatable.
    sampler = LatinHypercube(domain.dim, seed=seed)
    lhs_array = sampler.random(n=batch_size)

    # Rescaled into domain
    lhs_inputs = tuple([domain.scale(row) for row in lhs_array])

    return lhs_inputs


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
    to data using the fitted hyperparameters **from the supplied `gp`**, which avoids costly
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
    gp :
        A Gaussian process to calculate the normalised expected squared LOO errors for.
    domain :
        The domain of a simulator that the Gaussian process `gp` emulates. The data on
        which `gp` is trained are expected to have simulator inputs only from this domain.
    loo_errors_gp :
        Another Gaussian process that is trained on the LOO errors to
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
    multiplying the lengths of the domain's dimensions by ``sqrt(-0.5 / log(10**(-8)))``.
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
    gp :
        A Gaussian process to form the basis for the LOO GP.
    leave_out_idx :
        The index for the training datum of `gp` to leave out. This should be an index
        of the sequence returned by the ``gp.training_data`` property.
    loo_gp :
        Another Gaussian process that is trained on the LOO data and then
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

    Implements the PEI detailed in Mohammadi et. al. (2022)[1]. The functionality in this
    class applies to Gaussian processes that have been trained on data with inputs lying
    in the supplied simulator domain.

    If `additional_repulsion_pts` is provided, then these simulator inputs will be used as
    repulsion points when calculating pseudo-expected improvement of the LOO errors GP (in
    addition to the training inputs for the provided Gaussian process and the
    pseudopoints, which are always used as repulsion points). The additional
    repulsion points must belong to the simulator domain `domain`.

    Deep copies of the supplied `gp` and `domain` are stored internally upon initialisation of a new
    instance and subsequently used for calculations. This means that if the supplied `gp`
    or `domain` are modified after creation of a ``PEICalculator`` instance, then this
    won't affect the instance's behaviour. As a typical example, if a ``PEICalculator``
    instance ``pei`` is created with `gp` and `gp` is then trained on new data, then this
    won't be reflected in the calculation of pseudo-expected improvement, repulsion or
    expected improvement using ``pei``. To calculate these values for the updated ``gp``,
    a new ``PEICalculator`` instance would need to be created with the new version of
    ``gp``.

    Parameters
    ----------
    domain :
        The domain of a simulation.
    gp :
        A Gaussian process model, which is trained on data where the simulator inputs are
        in `domain`.
    additional_repulsion_pts :
        A collection of simulator inputs from `domain` that should be used
        as repulsion points when computing pseudo-expected improvement.

    Attributes
    ----------
    repulsion_points : tuple[Input]
        (Read-only) The current set of repulsion points used in calculations; as a tuple
        with no repeated elements.

    Raises
    ------
    ValueError
        If any of the new repulsion points don't belong to the supplied simulator domain.

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
    [1] Mohammadi, H. et al. (2022) "Cross-Validation-based Adaptive Sampling for
    Gaussian process models". DOI: <https://doi.org/10.1137/21M1404260>
    """

    def __init__(
        self,
        domain: SimulatorDomain,
        gp: AbstractGaussianProcess,
        additional_repulsion_pts: Optional[Collection[Input]] = None,
    ):
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

        self._domain = copy.deepcopy(domain)
        self._gp = copy.deepcopy(gp)
        self._validate_training_data()
        self._max_targets = self._calculate_max_targets()
        self._standard_norm = norm(loc=0, scale=1)

        # Initialise repulsion points
        training_inputs = [datum.input for datum in gp.training_data]
        self._other_repulsion_points = []
        self._add_repulsion_points(domain.calculate_pseudopoints(training_inputs))
        self._add_repulsion_points(
            self._parse_additional_repulsion_pts(additional_repulsion_pts, domain)
        )

    @staticmethod
    def _parse_additional_repulsion_pts(
        additional_repulsion_pts: Any, domain: SimulatorDomain
    ) -> list[Input]:

        if additional_repulsion_pts is None:
            return list()
        else:
            _check_collection_of_inputs(
                additional_repulsion_pts,
                name=f"{additional_repulsion_pts=}".split("=")[0],
            )
            if input_outside_domain := _find_input_outside_domain(
                additional_repulsion_pts, domain
            ):
                raise ValueError(
                    "Additional repulsion points must belong to simulator domain 'domain', "
                    f"but found input {input_outside_domain}."
                )
            else:
                return list(additional_repulsion_pts)

    @property
    def repulsion_points(self) -> tuple[Input]:
        """(Read-only) The current set of repulsion points used in calculations."""

        training_points = [datum.input for datum in self._gp.training_data]
        return tuple(self._other_repulsion_points + training_points)

    def _calculate_max_targets(self) -> Real:
        return max(self._gp.training_data, key=lambda datum: datum.output).output

    def _validate_training_data(self) -> None:
        if not self._gp.training_data:
            raise ValueError("Expected 'gp' to have nonempty training data.")

        if not all(isinstance(datum, TrainingDatum) for datum in self._gp.training_data):
            raise TypeError(
                f"Expected all elements in '_gp.training_data' to be of {type(TrainingDatum)},"
                f"but one or more elements were of an unexpected type."
            )

    def _validate_input_type(self, x: Any, method_name: str) -> Input:
        if not isinstance(x, Input):
            raise TypeError(
                f"In method '{method_name}', expected 'x' to be of type {Input}, "
                f"but received {type(x)} instead."
            )

        return x

    def compute(self, x: Input) -> Real:
        """
        Compute the pseudo-expected improvement (PEI) for a given input.

        Parameters
        ----------
        x :
            The simulator input to calculate PEI for.

        Returns
        -------
        Real
            The computed PEI value for the given input.

        Examples
        --------
        >>> input_point = Input(2.0, 3.0)
        >>> pei = pei_calculator.compute(input_point)

        >>> array_input = np.array([2.0, 3.0])
        >>> pei = pei_calculator.compute(array_input)

        Notes
        -----
        This method calculates the PEI at a given point `x`, which is the product of the
        expected improvement (EI) and the repulsion factor. The PEI is a metric used in
        Bayesian optimisation to balance exploration and exploitation, taking into account
        both the potential improvement over the current best target and the desire to
        explore less sampled regions of the domain.
        """

        return self.expected_improvement(x) * self.repulsion(x)

    def add_repulsion_points(self, repulsion_points: Collection[Input]) -> None:
        """
        Add simulator inputs to the set of repulsion points.

        Updates the internal set of repulsion points used in the repulsion factor
        calculation. The additional repulsion points must belong to the simulator domain
        for this object (i.e. ``self.domain``). Simulator inputs very positively
        correlated with repulsion points result in low repulsion values, whereas inputs
        very negatively correlated with repulsion points result in high repulsion values.

        Parameters
        ----------
        repulsion_points :
            The inputs to be added to the repulsion points set.

        Raises
        ------
        ValueError
            If any of the new repulsion points don't belong to this object's simulator
            domain.

        Examples
        --------
        >>> repulsion_point = Input(4.0, 5.0)
        >>> pei_calculator.add_repulsion_points([repulsion_point])

        >>> repulsion_points = [Input(4.0, 5.0), Input(4.1, 5.1)]
        >>> pei_calculator.add_repulsion_points(repulsion_points)
        """

        _check_collection_of_inputs(
            repulsion_points, name=f"{repulsion_points=}".split("=")[0]
        )

        if input_not_in_domain := _find_input_outside_domain(
            repulsion_points, self._domain
        ):
            raise ValueError(
                f"Repulsion points must belong to the simulator domain for this {__class__.__name__}, "
                f"but found input {input_not_in_domain}."
            )

        self._add_repulsion_points(repulsion_points)

        return None

    def _add_repulsion_points(self, repulsion_points: Collection[Input]) -> None:
        """Add new repulsion points (without arg validation); cf. add_repulsion_points."""

        for x in repulsion_points:
            if x not in self.repulsion_points:
                self._other_repulsion_points.append(x)

        return None

    def expected_improvement(self, x: Input) -> Real:
        """
        Calculate the expected improvement (EI) for a given input.

        If the standard deviation of the prediction is within the default
        tolerance ``exauq.core.numerics.FLOAT_TOLERANCE`` of 0 then the EI returned is 0.0.

        Parameters
        ----------
        x :
            The simulator input to calculate expected improvement for.

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
        This method computes the EI of the given input point `x` using the Gaussian
        process stored within this instance. EI is a measure used in Bayesian optimisation
        and is particularly useful for guiding the selection of points in the domain where
        the objective function should be evaluated next. It is calculated based on the
        model's prediction at `x`, the current maximum target value, and the standard
        deviation of the prediction.
        """

        validated_x = self._validate_input_type(x, "expected_improvement")
        prediction = self._gp.predict(validated_x)

        if equal_within_tolerance(prediction.standard_deviation, 0):
            return 0.0

        u = (prediction.estimate - self._max_targets) / prediction.standard_deviation

        cdf_u = self._standard_norm.cdf(u)
        pdf_u = self._standard_norm.pdf(u)

        return (
            prediction.estimate - self._max_targets
        ) * cdf_u + prediction.standard_deviation * pdf_u

    def repulsion(self, x: Input) -> Real:
        """
        Calculate the repulsion factor for a given simulator input.

        This method calculates a repulsion effect of a given point `x` in relation to
        other, stored repulsion points. It is calculated as the product of terms ``1 -
        correlation(x, rp)``, where ``rp`` is a repulsion point and the correlation is
        computed with the Gaussian process supplied at this object's initialisation. The
        repulsion factor approaches zero for inputs that tend towards repulsion points
        (and is equal to zero at repulsion points). This can be used to discourage the
        selection of points near already sampled locations, facilitating exploration of
        the input space.

        Parameters
        ----------
        x :
            The simulator input to calculate the repulsion factor for.

        Returns
        -------
        Real
            The repulsion factor for the given input.

        Examples
        --------
        >>> input_point = Input(1.5, 2.5)
        >>> repulsion_factor = pei_calculator.repulsion(input_point)

        >>> array_input = np.array([1.5, 2.5])
        >>> repulsion_factor = pei_calculator.repulsion(array_input)
        """

        validated_x = self._validate_input_type(x, "repulsion")

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
    additional_repulsion_pts: Optional[Collection[Input]] = None,
    loo_errors_gp: Optional[AbstractGaussianProcess] = None,
    seed: Optional[int] = None,
) -> tuple[Input]:
    """Compute a new batch of design points adaptively for a single-level Gaussian process.

    Implements the cross-validation-based adaptive sampling for emulators, as described in
    Mohammadi et. al. (2022)[1]. This involves computing a Gaussian process (GP) trained
    on normalised expected squared errors arising from a leave-one-out (LOO)
    cross-validation, then finding design points that maximise the pseudo-expected
    improvement of this LOO errors GP.

    If `additional_repulsion_pts` is provided, then these simulator inputs will be used as
    repulsion points when calculating pseudo-expected improvement of the LOO errors GP (in
    addition to pseudopoints, which are always used as repulsion points). The additional
    repulsion points must belong to the simulator domain `domain`. See ``PEICalculator``
    for further details on repulsion points and ``exauq.core.modelling.SimulatorDomain``
    for further details on pseudopoints.

    By default, a deep copy of the main GP supplied (`gp`) is trained on the leave-one-out
    errors. Alternatively, another ``AbstractGaussianProcess`` can be supplied that will
    be trained on the leave-one-out errors (and so modified in-place), allowing for the
    use of different Gaussian process settings (e.g. a different kernel function).

    If `seed` is provided, then this will be used when maximising the pseudo-expected
    improvement of the LOO errors GP (a sequence of seeds will be generated to find each new
    simulator input in the batch). Providing a seed does not necessarily mean calculation
    of the output design points is deterministic, as this also depends on computation of
    the LOO errors GP being deterministic.

    Parameters
    ----------
    gp :
        A Gaussian process to compute new design points for.
    domain :
        The domain of a simulator that the Gaussian process `gp` emulates. The data on
        which `gp` is trained are expected to have simulator inputs only from this domain.
    batch_size :
        The number of new design points to compute. Should be a positive
        integer.
    additional_repulsion_pts :
        A collection of simulator inputs from `domain` that should be used
        as repulsion points when computing pseudo-expected improvement.
    loo_errors_gp :
        Another Gaussian process that is trained on the LOO errors as part
        of the adaptive sampling method. If ``None`` then a deep copy of `gp` will be used
        instead.
    seed :
        A random number seed to use when maximising pseudo-expected
        improvement. If ``None`` then the maximisation won't be seeded.

    Returns
    -------
    tuple[Input]
        The batch of new design points.

    Raises
    ------
    ValueError
        If any of the training inputs in `gp` do not belong to the simulator domain
        `domain`.

    See Also
    --------
    [`compute_loo_errors_gp`][exauq.core.designers.compute_loo_errors_gp]:
        Computation of the leave-one-out errors Gaussian process.

    [`PEICalculator`][exauq.core.designers.PEICalculator] :
        Pseudo-expected improvement calculation.

    [`SimulatorDomain.calculate_pseudopoints`][exauq.core.modelling.SimulatorDomain.calculate_pseudopoints] :
        Calculation of pseudopoints for a collection of simulator inputs.

    [`GaussianProcessPrediction.nes_error`][exauq.core.modelling.GaussianProcessPrediction.nes_error] :
        Normalised expected squared error for a prediction from a Gaussian process.

    [`optimisation.maximise`][exauq.utilities.optimisation.maximise] :
        Global maximisation over a simulator domain, used on pseudo-expected improvement
        for the LOO errors GP.

    References
    ----------
    [1] Mohammadi, H. et al. (2022) "Cross-Validation-based Adaptive Sampling for
        Gaussian process models". DOI: <https://doi.org/10.1137/21M1404260>
    """
    if not isinstance(batch_size, int):
        raise TypeError(
            f"Expected 'batch_size' to be of type int, but received {type(batch_size)} instead."
        )

    if batch_size < 1:
        raise ValueError(
            f"Expected batch size to be a positive integer, but received {batch_size} instead."
        )

    if additional_repulsion_pts is not None:
        _check_collection_of_inputs(
            additional_repulsion_pts, name=f"{additional_repulsion_pts=}".split("=")[0]
        )
        if input_outside_domain := _find_input_outside_domain(
            additional_repulsion_pts, domain
        ):
            raise ValueError(
                "Additional repulsion points must belong to simulator domain 'domain', "
                f"but found input {input_outside_domain}."
            )

    try:
        gp_e = compute_loo_errors_gp(gp, domain, loo_errors_gp=loo_errors_gp)
    except (TypeError, ValueError) as e:
        raise e from None

    pei = PEICalculator(domain, gp_e, additional_repulsion_pts=additional_repulsion_pts)

    seeds = generate_seeds(seed, batch_size)

    design_points = []
    for design_pt_seed in seeds:
        new_design_point, _ = maximise(
            lambda x: pei.compute(x), domain, seed=design_pt_seed
        )
        design_points.append(new_design_point)
        pei.add_repulsion_points([new_design_point])

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
    mlgp :
        A multi-level Gaussian process to form the basis for the multi-level LOO GP.
    level :
        The level containing the datum to leave out.
    leave_out_idx :
        The index for the training datum of `mlgp` to leave out, at the level `level`.
    loo_gp :
        A Gaussian process that is trained on the LOO data and then
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
    fitted hyperparameters **from the supplied Gaussian process `gp`**.

    By default, the LOO GP used will be a deep copy of `gp` trained on the leave-one-out
    data. Alternatively, another ``AbstractGaussianProcess`` can be supplied that will be
    trained on the leave-one-out data. This can be more efficient when repeated
    calculation of a LOO GP is required.

    Parameters
    ----------
    gp :
        A Gaussian process to form the basis for the LOO GP.
    leave_out_idx :
        The index for the training datum of `gp` to leave out. This should be an index
        of the sequence returned by the ``gp.training_data`` property.
    loo_gp :
        Another Gaussian process that is trained on the LOO data and then
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
    gp :
        A Gaussian process.
    x :
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

        mean = float(gp.covariance_matrix([x]) @ gp.kinv @ training_outputs)

    except ValueError as e:
        if not training_inputs:
            raise ValueError(
                "Cannot calculate zero-mean prediction: 'gp' hasn't been trained on any data."
            )
        else:
            raise ValueError(f"Cannot calculate zero-mean prediction: {e}")
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
    mlgp :
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
    [`compute_multi_level_loo_prediction`][exauq.core.designers.compute_multi_level_loo_prediction] :
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
    mlgp :
        A multi-level GP to calculate the normalised expected squared LOO errors for.
    domain :
        The domain of a simulator that the multi-level Gaussian process `mlgp` emulates.
        The data on which each level of `mlgp` is trained are expected to have simulator
        inputs only from this domain.
    output_mlgp :
        Another multi-level GP that is trained on the LOO errors to
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
    [`compute_multi_level_loo_error_data`][exauq.core.designers.compute_multi_level_loo_error_data] :
        Calculation of the leave-one-out error data used to train this function's output.

    Notes
    -----
    The lower bounds on the correlation length scale parameters are obtained by
    multiplying the lengths of the domain's dimensions by ``sqrt(-0.5 / log(10**(-8)))``.
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
    additional_repulsion_pts: Optional[MultiLevel[Collection[Input]]] = None,
    seeds: Optional[MultiLevel[int]] = None,
) -> MultiLevel[tuple[Input]]:
    """Compute a batch of design points adaptively for a multi-level Gaussian process (GP).

    Implements the cross-validation-based adaptive sampling for multi-level Gaussian
    process models, as described in Kimpton et. al. (2023)[1]. This involves computing a
    multi-level GP that is trained on normalised expected squared errors arising from a
    multi-level leave-one-out (LOO) cross-validation. The design points returned are those
    that maximise weighted pseudo-expected improvements (PEIs) of this multi-level LOO
    errors GP across levels, where the PEIs are weighted according to the costs of
    computing the design points on simulators at the levels.

    The `costs` should represent the successive differences costs of running each level's
    simulator on a single input. For example, if the level costs were 1, 10, 100 for levels
    1, 2, 3 respectively, then 1, 11, 110 would need to be supplied if successive differences
    was the chosen method for calculating costs.

    If `additional_repulsion_pts` is provided, then these points will be added into the
    calculations at the level they are allocated to in the PEI.

    If `seeds` is provided, then the seeds provided for the levels will be used when
    maximising the pseudo-expected improvement of the LOO errors GP for each level (a
    sequence of seeds will be generated level-wise to find each new simulator input
    in the batch). Note that ``None`` can be provided for a level, which means the maximisation
    at that level won't be seeded. Providing seeds does not necessarily mean calculation of the
    output design points is deterministic, as this also depends on computation of the LOO
    errors GP being deterministic.

    The adaptive sampling method assumes that none of the levels in the multi-level GP
    share common training simulator inputs; a ValueError will be raised if this is not the
    case.

    Parameters
    ----------
    mlgp :
        The multi-level GP to create the design points for.
    domain :
        The domain of a simulator that the multi-level Gaussian process `mlgp` emulates.
        The data on which each level of `mlgp` is trained are expected to have simulator
        inputs only from this domain.
    costs :
        The costs of running a simulation at each of the levels.
    batch_size :
        The number of new design points to compute. Should be a positive
        integer.
    additional_repulsion_pts:
        A multi-level collection of hand-chosen Input repulsion points to
        aid computation of samples.
    seeds :
        A multi-level collection of random number seeds to use when
        maximising pseudo-expected improvements for each level. If ``None`` then none of
        the maximisations will be seeded.

    Returns
    -------
    MultiLevel[tuple[Input]]
        A MultiLevel tuple of inputs containing all of the new design points at the correct level

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
    [`compute_multi_level_loo_errors_gp`][exauq.core.designers.compute_multi_level_loo_errors_gp]:
        Computation of the multi-level leave-one-out errors GP.

    [`PEICalculator`][exauq.core.designers.PEICalculator] :
        Pseudo-expected improvement calculation.

    [`GaussianProcessPrediction.nes_error`][exauq.core.modelling.GaussianProcessPrediction.nes_error]:
        Normalised expected squared error for a prediction from a Gaussian process.

    [`optimisation.maximise`][exauq.utilities.optimisation.maximise] :
        Global maximisation over a simulator domain, used on pseudo-expected improvement
        for the multi-level LOO errors GP.

    References
    ----------
    [1] Kimpton, L. M. et al. (2023) "Cross-Validation Based Adaptive Sampling for
        Multi-Level Gaussian Process Models". arXiv: <https://arxiv.org/abs/2307.09095>
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

    if additional_repulsion_pts is None:
        additional_repulsion_pts = MultiLevel({level: None for level in mlgp.levels})
    elif not isinstance(additional_repulsion_pts, MultiLevel):
        raise TypeError(
            f"Expected 'additional_repulsion_pts' to be of type MultiLevel collection of {Input}s, "
            f"but received {type(additional_repulsion_pts)} instead."
        )

    if seeds is None:
        seeds = MultiLevel({level: None for level in mlgp.levels})
    elif not isinstance(seeds, MultiLevel):
        raise TypeError(
            f"Expected 'seeds' to be of type {MultiLevel} of int, but "
            f"received {type(seeds)} instead."
        )

    if missing_levels := sorted(set(mlgp.levels) - set(seeds.levels)):
        raise ValueError(
            f"Level {missing_levels[0]} from 'mlgp' does not have associated level "
            "from 'seeds'."
        )

    if not isinstance(batch_size, int):
        raise TypeError(
            f"Expected 'batch_size' to be of type int, but received {type(batch_size)} instead."
        )

    if batch_size < 1:
        raise ValueError(
            f"Expected batch size to be a positive integer, but received {batch_size} instead."
        )

    # Generate seed sequences
    seeds = MultiLevel(
        {level: generate_seeds(seeds[level], batch_size) for level in mlgp.levels}
    )

    # Create LOO errors GP for each level
    ml_errors_gp = compute_multi_level_loo_errors_gp(mlgp, domain, output_mlgp=None)

    # Get the PEI calculator for each level
    ml_pei = ml_errors_gp.map(
        lambda level, gp: PEICalculator(
            domain, gp, additional_repulsion_pts=additional_repulsion_pts[level]
        )
    )

    # Create empty dictionary for levels and design points
    design_points = defaultdict(list)

    # Iterate through batch - recalculating levels and design points
    for i in range(batch_size):

        # Calculate new level by maximising PEI
        maximal_pei_values = ml_pei.map(
            lambda level, pei: maximise(
                lambda x: pei.compute(x) / costs[level],
                domain,
                seed=seeds[level][i],
            )
        )
        level, (new_design_pt, _) = max(
            maximal_pei_values.items(), key=lambda item: item[1][1]
        )

        # Reset PEI and add calculated design pt to repulsion points.
        pei = ml_pei[level]
        pei.add_repulsion_points([new_design_pt])

        # Add design inputs to dictionary level
        design_points[level].append(new_design_pt)

    return MultiLevel({level: tuple(inputs) for level, inputs in design_points.items()})


def create_data_for_multi_level_loo_sampling(
    data: MultiLevel[Sequence[TrainingDatum]],
    correlations: Union[MultiLevel[Real], Real] = 1,
) -> MultiLevel[Sequence[TrainingDatum]]:
    """Prepare data from the simulators to be ready for multi-level adaptive sampling.

    For the implementation of creating the successive simulator differences used by
    `compute_multi_level_loo_samples`, the data needs to satisfy the correct delta
    calculations calculated between level $L$ and $L-1$ for the same inputs.
    It must then be ensured that there are no repeats of inputs across the different
    levels and hence are removed from the data as it makes no mathematical sense to have
    repeated inputs across the multi-level GP with differing outputs.

    Parameters
    ----------
    data:
        Training data for the simulator at that level.
    correlations:
        The Markov-like correlations between simulators at successive levels
    Returns
    -------
    MultiLevel[Sequence[TrainingDatum]
        A MultiLevel Sequence of TrainingDatum recalculated with deltas calculated
        and repeated inputs across levels removed.

    """

    if not isinstance(data, MultiLevel):
        raise TypeError(
            "Expected 'data' to be of type MultiLevel Sequence of TrainingDatum, "
            f"but received {type(data)} instead."
        )

    if not data.items():
        warn("'data' passed was empty and therefore no transformations taken place.")
        return data

    if correlations is not None:
        if not isinstance(correlations, MultiLevel) and not isinstance(
            correlations, Real
        ):
            raise TypeError(
                "Expected 'correlations' to be of type MultiLevel Real or Real, "
                f"but received {type(correlations)} instead."
            )

    top_level = max(data.levels)
    bottom_level = min(data.levels)
    if isinstance(correlations, Real):
        correlations = MultiLevel(
            {level: correlations for level in data.levels if level < top_level}
        )

    delta_data = MultiLevel({level: [] for level in data.levels})

    for level in reversed(data.levels):
        if level != bottom_level:

            if max(correlations.levels) < top_level - 1:
                raise ValueError(
                    f"'Correlations' MultiLevel expected to be provided for at least max level of 'data' - 1: {top_level - 1}, but "
                    f"is only of length: {max(correlations.levels)}."
                )

            # Catch missing levels in data MultiLevel
            try:
                prev_level_data = data[level - 1]

            except KeyError:
                continue

            for datum in data[level]:
                # Find datum in previous level with the same input as the current datum
                prev_level_datum_list = [
                    dat
                    for dat in prev_level_data
                    if equal_within_tolerance(dat.input, datum.input)
                ]

                if prev_level_datum_list:
                    prev_level_datum = prev_level_datum_list[0]

                    # Remove matching inputs from the rest of data
                    data = _remove_multi_level_repeated_input(data, datum, level)

                    # Compute output for this input as the difference between this level's
                    # output and the previous level's output multiplied by correlation.
                    delta_output = (
                        datum.output - correlations[level - 1] * prev_level_datum.output
                    )

                    # Add input and the computed output to the training data to return
                    delta_data[level].append(TrainingDatum(datum.input, delta_output))

        else:
            # In the case of the bottom level simply return raw values
            delta_data[level] = data[level]

    for lvl in delta_data.levels:
        if not delta_data[lvl]:
            warn(f"After processing, Level {lvl} is empty. Check your input data")

    return delta_data


def compute_delta_coefficients(
    levels: Union[Optional[Sequence[int]], int],
    correlations: Union[MultiLevel[Real], Real] = 1,
) -> MultiLevel[Real]:
    """Calculate the delta coefficients from the Markov-like correlations.

    The levels argument creates the correlations for the number of levels that there are. If a sequence is passed
    then this is expected to be a range of levels for the mlgp. Optionally, you can simply pass the number of levels
    as an integer and this will create the correlations up to that level.

    By default the constant correlation of 1 is applied to every level if no correlation is
    provided. Note that the correlations only run to level $L - 1$ as it denotes the correlation
    between that level and the one above. If a 'Real' value is provided, then this value is provided
    for every level in the multi-level correlations object.

    Parameters
    ----------
    levels:
        The number of levels or a tuple of levels for the coefficients to be calculated for.

    correlations:
        The Markov-like correlations between simulators at successive levels.

    Returns
    -------
    MultiLevel[Real]:
        The delta coefficients calculated from the correlations across levels.
    """

    if not isinstance(levels, int) and not isinstance(levels, Sequence):
        raise TypeError(
            "Expected 'levels' to be of type Sequence of int or int, "
            f"but received {type(levels)} instead."
        )

    if isinstance(levels, int):
        levels = range(1, levels + 1)

    if not all(isinstance(level, int) for level in levels):
        raise TypeError(
            "Expected 'levels' to be of type Sequence of int or int, "
            f"but received unexpected types."
        )

    if correlations is not None:
        if not isinstance(correlations, MultiLevel) and not isinstance(
            correlations, Real
        ):
            raise TypeError(
                "Expected 'correlations' to be of type MultiLevel Real or Real, "
                f"but received {type(correlations)} instead."
            )

    if isinstance(correlations, Real):
        correlations = MultiLevel(
            {level: correlations for level in levels if level < max(levels)}
        )

    delta_coefficients = MultiLevel(
        {max(levels): 1}
        | correlations.map(
            lambda k, _: math.prod(
                correlations[level] for level in correlations.levels if level >= k
            )
        )
    )

    return delta_coefficients


def _remove_multi_level_repeated_input(
    data: MultiLevel[Sequence[TrainingDatum]],
    datum: TrainingDatum,
    level: int,
) -> MultiLevel[Sequence[TrainingDatum]]:
    """Remove a repeated input from a set of training data at a specific level.

    Currently a utility function to `create_data_for_multi_level_loo_sampling` to improve readability,
    although it may become a more generalised function at a later date. It will remove all repeated
    entries across the multilevel for the input of a single datum.

    Passing the level ensures that you keep the highest fidelity level of that repeated input.

    Parameters
    ----------
    data:
        Training data to be checked for repeated input
    datum:
        Single TrainingDatum used for comparison of input.
    level:
        The highest fidelity level at which the input appears.

    Returns
    -------
    MultiLevel[Sequence[TrainingDatum]]
        Returns an updated version of data with repeated inputs removed.
    """

    for lvl in data.levels:
        if lvl <= level:
            data[lvl] = [
                dat
                for dat in data[lvl]
                if not equal_within_tolerance(dat.input, datum.input)
            ]

    # Re-enter the original comparing TrainingDatum
    data[level].append(datum)
    return data
