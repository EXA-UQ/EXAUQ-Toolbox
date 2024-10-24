"""Contains fakes used to support unit tests
"""

from __future__ import annotations

import dataclasses
from collections.abc import Collection, Sequence
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from exauq.core.modelling import (
    AbstractGaussianProcess,
    AbstractSimulator,
    GaussianProcessHyperparameters,
    GaussianProcessPrediction,
    Input,
    OptionalFloatPairs,
    TrainingDatum,
)
from exauq.sim_management.hardware import HardwareInterface
from exauq.sim_management.simulators import SimulationsLog


class WhiteNoiseGP(AbstractGaussianProcess):
    """A Gaussian process emulator with a white noise kernel.

    The 'white noise' kernel is the function that takes in a pair of simulator inputs
    ``x1, x2`` and returns a positive constant ``noise_level`` if ``x1 == x2`` and zero
    otherwise. The process variance for this GP is equal to the noise level.

    Parameters
    ----------
    prior_mean : float, default 0
        The prior mean value for the GP at simulator inputs (effectively a constant
        function).
    noise_level : float, default 1
        The noise level defining the kernel function (equivalently, the process variance).

    Attributes
    ----------
    predictive_mean : float
        The prior mean value for the GP at simulator inputs.
    noise_level : float
        The noise level used to define the kernel function (equivalently, the process
        variance).
    training_data : tuple[TrainingDatum]
        Defines the pairs of inputs and simulator outputs on which the emulator
        has been trained.
    fit_hyperparameters : WhiteNoiseGPHyperparameters or None
        The hyperparameters of the fit for this GP, or ``None`` if this emulator
        has not been fitted to data. Note that the correlation length scale parameters and
        nugget are irrelevant in this context and are not used by this GP.
    hyperparameter_bounds : Sequence[OptionalFloatPairs] or None
        The hyperparameter bounds that were supplied when fitting this emulator to
        data, or ``None`` if none were. Note that the bounds for the correlation length
        scale parameters are irrelevant in this context.
    """

    def __init__(self, prior_mean: float = 0, noise_level: float = 1):
        super()
        self._training_data = tuple()
        self._prior_mean = prior_mean
        self._noise_level = noise_level
        self._fit_hyperparameters = None
        self._hyperparameter_bounds = None
        self._kinv = np.array([])

    @property
    def prior_mean(self) -> float:
        """The prior mean value for the GP at simulator inputs."""

        return self._prior_mean

    @property
    def noise_level(self) -> float:
        """The noise level used to define the kernel function (equivalently, the process
        variance)."""

        return self._noise_level

    @property
    def training_data(self) -> tuple[TrainingDatum]:
        """Get the data on which the emulator has been trained."""

        return self._training_data

    @property
    def fit_hyperparameters(self) -> WhiteNoiseGPHyperparameters:
        """The hyperparameters of the fit for this emulator, or ``None`` if this emulator
        has not been fitted to data. Note that the correlation length scale parameters and
        nugget are irrelevant in this context and are not used by this GP."""

        return self._fit_hyperparameters

    @property
    def hyperparameter_bounds(self) -> Optional[Sequence[OptionalFloatPairs]]:
        """The hyperparameter bounds that were supplied when fitting this emulator to
        data, or ``None`` if none were. Note that the bounds for the correlation length
        scale parameters are irrelevant in this context."""

        return self._hyperparameter_bounds
    
    @property
    def kinv(self) -> NDArray:
        """The inverse of the covariance matrix, k, for the training data, or an empty NumPy array if 
        this emulator has not been fitted to data."""

        return self._kinv

    def fit(
        self,
        training_data: Collection[TrainingDatum],
        hyperparameters: Optional[WhiteNoiseGPHyperparameters] = None,
        hyperparameter_bounds: Optional[Sequence[OptionalFloatPairs]] = None,
    ) -> None:
        """Fits the GP on the given data.

        The effect of fitting this GP to the training data is to update the mean function
        so that the training outputs are returned at training inputs, while the prior
        mean is still returned at inputs not in the training data. The kernel function
        is also updated so that it returns zero when applied to pairs of training data
        inputs (i.e. ``kernel(x, x) == 0`` for training inputs ``x``).

        If no hyperparameters are provided then the kernel function will continue to use
        the noise level supplied at this GP's initialisation, so long at this falls
        within any supplied bounds given in `hyperparameter_bounds`. (If the noise level
        falls outside of the bounds, then kernel will instead use the value of the bound
        nearest the noise level.)

        Parameters
        ----------
        training_data : Collection[TrainingDatum]
            Defines the pairs of inputs and simulator outputs on which to train
            this GP.
        hyperparameters : WhiteNoiseGPHyperparameters, optional
            (Default: ``None``) If not ``None`` then this should define the
            process variance that will be used as the new noise level in the kernel
            function (the other hyperparameters are not used in this context). If ``None``
            then the noise level supplied at the creation of this GP will continue to be
            used, subject to any bounds supplied in `hyperparameter_bounds`.
        hyperparameter_bounds : sequence of tuple[Optional[float], Optional[float]], optional
            (Default: ``None``) If not ``None``, then the bounds on the process variance
            (i.e. noise level) will be respected if fitting this GP without a specified
            collection of hyperparameters.
        """

        self._training_data = tuple(training_data)
        if hyperparameters is not None:
            self._fit_hyperparameters = hyperparameters
            
        elif hyperparameter_bounds is not None:
            lower, upper = hyperparameter_bounds[-1]
            if lower is not None and upper is not None and lower > upper:
                raise ValueError(
                    "Process variance upper bound must be greater than lower bound"
                )
            elif lower is not None and self._noise_level <= lower:
                process_var = lower
            elif upper is not None and self._noise_level >= upper:
                process_var = upper
            else:
                process_var = self._noise_level

            self._fit_hyperparameters = WhiteNoiseGPHyperparameters(
                process_var=process_var
            )
            self._hyperparameter_bounds = tuple(hyperparameter_bounds)
            
        else:
            self._fit_hyperparameters = WhiteNoiseGPHyperparameters(
                process_var=self._noise_level
            )
            self._hyperparameter_bounds = None
        
        self._kinv = self._compute_kinv()
        return None

    def _indicator_fn(self, x1: Input, x2: Input) -> int:
        """Returns 1 if the two inputs are equal and zero otherwise."""

        if x1 == x2:
            return 1
        else:
            return 0

    def _kernel(self, x1: Input, x2: Input) -> float:
        """The (prior) kernel function."""

        return self._noise_level * self._indicator_fn(x1, x2)

    def correlation(self, inputs1: Sequence[Input], inputs2: Sequence[Input]) -> NDArray:
        """Compute the (prior) correlation matrix for two sequences of simulator inputs.

        The matrix entry at position `(i, j)` is equal to 1 if
        ``inputs1[i] == inputs2[j]`` and zero otherwise.

        Parameters
        ----------
        inputs1, inputs2 : Sequence[Input]
            Sequences of simulator inputs.

        Returns
        -------
        numpy.ndarray
            The correlation matrix for the two sequences of inputs, as a 2-dim Numpy array
            of shape ``(len(inputs1), len(inputs2))``.
        """

        return np.array(
            tuple(tuple(self._indicator_fn(xi, xj) for xj in inputs2) for xi in inputs1)
        )

    def predict(self, x: Input) -> GaussianProcessPrediction:
        """Estimate the simulator output for a given input.

        The emulator will predict the correct simulator output, with zero variance, for
        `x` which feature in the training data. For new `x`, the prediction will be equal
        to prior mean supplied at this GP's initialisation and its variance will be equal
        to the fitted noise level (i.e. the fitted process variance).

        Parameters
        ----------
        x : Input
            An input to a simulator.

        Returns
        -------
        GaussianProcessPrediction
            This GP's prediction of the simulator output from the given the input.
        """
        if not isinstance(x, Input):
            raise TypeError

        for datum in self._training_data:
            if datum.input == x:
                return GaussianProcessPrediction(estimate=datum.output, variance=0)

        return GaussianProcessPrediction(
            estimate=self._prior_mean, variance=self.fit_hyperparameters.process_var
        )


class WhiteNoiseGPHyperparameters(GaussianProcessHyperparameters):
    def __init__(self, process_var: float):
        super().__init__(corr_length_scales=[1], process_var=process_var)


class FakeGP(AbstractGaussianProcess):
    """A concrete, fake Gaussian process emulator for emulating 1-dimensional simulators.

    Simulator outputs on which the emulator has been fitted are predicted correctly with
    zero predictive variance. Inputs that have not been fitted to the emulator are
    predicted as zero with a variance defined by a hyperparameter.

    Parameters
    ----------
    predictive_mean : float, default 0
        The mean value that this emulator should predict away from training inputs.

    Attributes
    ----------
    predictive_mean : float
        The mean value that this emulator predicts away from training inputs.
    training_data : tuple[TrainingDatum]
        Defines the pairs of inputs and simulator outputs on which the emulator
        has been trained. Each `TrainingDatum` should have a 1-dim
        `Input`.
    fit_hyperparameters : FakeGPHyperparameters or None
        The hyperparameters of the fit for this emulator, or ``None`` if this emulator
        has not been fitted to data.
    hyperparameter_bounds : Sequence[OptionalFloatPairs] or None
        The hyperparameter bounds that were supplied when fitting this emulator to
        data, or ``None`` if none were.
    """

    def __init__(self, predictive_mean: float = 0):
        super()
        self._training_data = tuple()
        self._predictive_mean = predictive_mean
        self._predictive_variance = 1
        self._fit_hyperparameters = None
        self._hyperparameter_bounds = None
        self._kinv = np.array([])

    @property
    def predictive_mean(self) -> float:
        """Get the mean value that this emulator predicts away from training inputs."""

        return self._predictive_mean

    @property
    def training_data(self) -> tuple[TrainingDatum]:
        """Get the data on which the emulator has been trained."""

        return self._training_data

    @property
    def fit_hyperparameters(self) -> FakeGPHyperparameters:
        """The hyperparameters of the fit for this emulator, or ``None`` if this emulator
        has not been fitted to data."""

        return self._fit_hyperparameters

    @property
    def hyperparameter_bounds(self) -> Optional[Sequence[OptionalFloatPairs]]:
        """The hyperparameter bounds that were supplied when fitting this emulator to
        data, or ``None`` if none were."""

        return self._hyperparameter_bounds
    
    @property
    def kinv(self) -> NDArray:
        """(Read-only) The inverse of the covariance matrix of the training data, 
        or an empty NumPy array if the model has not been fitted to data. Note
        in this test GP this is not used."""
         
        return self._kinv

    def fit(
        self,
        training_data: Collection[TrainingDatum],
        hyperparameters: Optional[FakeGPHyperparameters] = None,
        hyperparameter_bounds: Optional[Sequence[OptionalFloatPairs]] = None,
    ) -> None:
        """Fits the emulator on the given data.

        Any prior training data are discarded, so that each call to `fit` effectively fits
        the emulator again from scratch. If hyperparameters are provided then the variance
        for predictions of unseen inputs will be set to the variance hyperparameter,
        otherwise a default variance of 1 will be set.

        Parameters
        ----------
        training_data : Collection[TrainingDatum]
            Defines the pairs of inputs and simulator outputs on which to train
            the emulator. Each `TrainingDatum` should have a 1-dim `Input`.
        hyperparameters : FakeGPHyperparameters, optional
            (Default: ``None``) If not ``None`` then this should define the variance for
            predictions away from inputs defined in `training_data`.
        hyperparameter_bounds : sequence of tuple[Optional[float], Optional[float]], optional
            (Default: ``None``) Not used.
        """

        self._training_data = tuple(training_data)
        self._predictive_variance = (
            hyperparameters.process_var if hyperparameters is not None else 1
        )
        self._fit_hyperparameters = hyperparameters
        self._hyperparameter_bounds = (
            tuple(hyperparameter_bounds) if hyperparameter_bounds is not None else None
        )
        
        return None

    def correlation(
        self, inputs1: Sequence[Input], inputs2: Sequence[Input]
    ) -> tuple[tuple[float, ...], ...]:
        """Compute the correlation matrix for two sequences of simulator inputs.

        This just returns a matrix where each entry is equal to 0.5.

        Parameters
        ----------
        inputs1, inputs2 : Sequence[Input]
            Sequences of simulator inputs.

        Returns
        -------
        tuple[tuple[float, ...], ...]
            The correlation matrix for the two sequences of inputs. The outer tuple
            consists of ``len(inputs1)`` tuples of length ``len(inputs2)``.
        """

        return tuple(tuple(0.5 for xj in inputs2) for xi in inputs1)

    def predict(self, x: Input) -> GaussianProcessPrediction:
        """Estimate the simulator output for a given input.

        The emulator will predict the correct simulator output for `x` which
        feature in the training data. For new `x`, the prediction with have as its mean
        value `predictive_mean` and its variance `predictive_variance`.

        Parameters
        ----------
        x : Input
            An input to the simulator.

        Returns
        -------
        GaussianProcessPrediction
            This GP's prediction of the simulator output from the given the input.
        """
        if not isinstance(x, Input):
            raise TypeError

        for datum in self._training_data:
            if datum.input == x:
                return GaussianProcessPrediction(estimate=datum.output, variance=0)

        return GaussianProcessPrediction(
            estimate=self._predictive_mean, variance=self._predictive_variance
        )


@dataclasses.dataclass(frozen=True)
class FakeGPHyperparameters(GaussianProcessHyperparameters):
    # Override superclass __post_init__ to disable arg validation. This allows for
    # creating hyperparameter values that can test edge cases.
    def __post_init__(self):
        pass


class OneDimSimulator(AbstractSimulator):
    """A basic simulator defined on a one-dimensional domain.

    This simulator simply defines the identity function ``f(x) = x`` and is
    defined on a closed interval [a, b].

    Parameters
    ----------
    lower_limit: float
        The lower limit of the domain on which the simulator is defined.
    upper_limit: float
        The upper limit of the domain on which the simulator is defined.

    Attributes
    ----------
    domain: OneDimDomain
        The domain on which the simulator is defined.
    """

    def __init__(self, lower_limit: float, upper_limit: float):
        self.lower_limit: float = lower_limit
        self.upper_limit: float = upper_limit

    def compute(self, x: Input) -> float:
        """Evaluate the identity function at the given point.

        Parameters
        ----------
        x : Input
            The input at which to evaluate the simulator.

        Returns
        -------
        float
            The value of the input `x`.
        """
        return x.value


class DumbHardwareInterface(HardwareInterface):
    """A stub for a hardware interface that doesn't do anything beyond the implementations
    in the abstract HardwareInterface class."""

    def __init__(self, name: str = "DumbInterface"):
        super().__init__(name)

    def submit_job(self, job) -> None:
        return super().submit_job(job)

    def get_job_status(self, job_id) -> None:
        return super().get_job_status(job_id)

    def get_job_output(self, job_id) -> None:
        return super().get_job_output(job_id)

    def cancel_job(self, job_id) -> None:
        return super().cancel_job(job_id)

    def wait_for_job(self, job_id) -> None:
        return super().wait_for_job(job_id)


class DumbJobManager:
    """A fake job manager that simply records new simulations in the log.

    Parameters
    ----------
    simulations_log : SimulationsLog
        The log to record simulations to.
    *args : tuple
        Additional arguments that may be required in constructing a JobManager instance.
    """

    def __init__(self, simulations_log: SimulationsLog, *args):
        self._simulations_log = simulations_log

    def submit(self, x: Input) -> None:
        """Submit an input to be recorded in the log.

        This is intended to mock out submission of inputs for computation by a simulator.

        Parameters
        ----------
        x : Input
            The simulator input to record.
        """
        self._simulations_log.add_new_record(x)
