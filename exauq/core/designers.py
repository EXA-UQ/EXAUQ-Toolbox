import copy
import math
from collections.abc import Collection
from typing import Optional

import numpy as np
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
    def __init__(self, domain: SimulatorDomain, gp: AbstractGaussianProcess):
        """
        Initialises the PEI (PseudoExpected Improvement) Calculator.

        :param domain: An instance of SimulatorDomain, representing the domain of simulation.
        :param gp: An instance of AbstractGaussianProcess, representing the Gaussian process model.
        """
        # TODO: Implement initialisation logic
        raise NotImplementedError("Initialisation not yet implemented.")

    def compute(self, x: Input) -> float:
        """
        Computes the PEI based on the given input.

        :param x: An instance of Input, representing the input data.
        :return: A float value representing the computed PEI.
        """
        # TODO: Implement computation logic
        raise NotImplementedError("Computation method not yet implemented.")


def pei(x: Input, gp: AbstractGaussianProcess) -> float:
    raise NotImplementedError


def expected_improvement(x: Input, gp: AbstractGaussianProcess) -> float:
    prediction = gp.predict(x)
    if equal_within_tolerance(prediction.variance, 0):
        return 0.0

    # This will end up being calculated for each point... maybe a class would be more efficient
    max_targets = max(gp.training_data, key=lambda datum: datum.output).output

    u = (prediction.estimate - max_targets) / math.sqrt(prediction.variance)

    return (prediction.estimate - max_targets) * norm(loc=0, scale=1).cdf(u) + math.sqrt(
        prediction.variance
    ) * norm(loc=0, scale=1).pdf(u)


def repulsion(x: Input, gp: AbstractGaussianProcess) -> np.array:
    pass


def compute_single_level_loo_samples(
    gp: AbstractGaussianProcess,
    domain: SimulatorDomain,
    batch_size: int = 1,
    loo_errors_gp: Optional[AbstractGaussianProcess] = None,
) -> tuple[Input]:
    gp_e = compute_loo_errors_gp(gp)

    return maximise(lambda x: pei(x, gp_e), domain)
