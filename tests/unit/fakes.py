"""Contains fakes used to support unit tests
"""
from __future__ import annotations

import dataclasses
from collections.abc import Collection, Sequence
from typing import Optional

from exauq.core.hardware import HardwareInterface
from exauq.core.modelling import (
    AbstractGaussianProcess,
    AbstractHyperparameters,
    AbstractSimulator,
    Input,
    OptionalFloatPairs,
    Prediction,
    TrainingDatum,
)
from exauq.core.simulators import SimulationsLog


class FakeGP(AbstractGaussianProcess):
    """A concrete, fake Gaussian process emulator for emulating 1-dimensional simulators.

    Simulator outputs on which the emulator has been fitted are predicted correctly with
    zero predictive variance. Inputs that have not been fitted to the emulator are
    predicted as zero with a variance defined by a hyperparameter.


    Attributes
    ----------
    training_data: tuple[TrainingDatum]
        Defines the pairs of inputs and simulator outputs on which the emulator
        has been trained. Each `TrainingDatum` should have a 1-dim
        `Input`.
    """

    def __init__(self):
        super()
        self._training_data = tuple()

    @property
    def training_data(self) -> tuple[TrainingDatum]:
        """Get the data on which the emulator has been trained."""

        return self._training_data

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
        hyperparameters : DumbEmulatorHyperparameters, optional
            (Default: ``None``) If not ``None`` then this should define the variance for
            predictions away from inputs defined in `training_data`.
        hyperparameter_bounds : sequence of tuple[Optional[float], Optional[float]], optional
            (Default: ``None``) Not used.
        """

        self._training_data = tuple(training_data)
        self._predictive_variance = (
            hyperparameters.var if hyperparameters is not None else 1
        )

    def predict(self, x: Input) -> Prediction:
        """Estimate the simulator output for a given input.

        The emulator will predict the correct simulator output for `x` which
        feature in the training data. For new `x`, zero will be returned along with
        a variance that was specified when training.

        Parameters
        ----------
        x : Input
            An input to the simulator.

        Returns
        -------
        Prediction
            The emulator's prediction of the simulator output from the given the input.
        """
        for datum in self._training_data:
            if datum.input == x:
                return Prediction(estimate=datum.output, variance=0)

        return Prediction(estimate=0, variance=self._predictive_variance)


@dataclasses.dataclass(frozen=True)
class FakeGPHyperparameters(AbstractHyperparameters):
    var: float


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
