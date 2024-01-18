import copy
from collections.abc import Collection

import numpy as np

from exauq.core.modelling import (
    AbstractEmulator,
    AbstractGaussianProcess,
    Input,
    SimulatorDomain,
    TrainingDatum,
)
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
            self._esloo_errors = [compute_nes_loo_error(emulator, leave_out_idx=1)] * len(
                emulator.training_data
            )

        return [Input(1)] * size

    @property
    def esloo_errors(self):
        return self._esloo_errors


def compute_nes_loo_error(gp: AbstractGaussianProcess, leave_out_idx: int) -> float:
    training_data = list(gp.training_data)
    try:
        left_out_datum = training_data.pop(leave_out_idx)
    except IndexError:
        if len(training_data) == 0:
            raise ValueError(
                "Cannot compute leave one out error with 'gp' because it has not been "
                "trained on data."
            ) from None
        else:
            raise ValueError(
                f"Leave out index {leave_out_idx} is not within the bounds of the training "
                "data for 'gp'."
            ) from None

    loo_emulator = copy.copy(gp)
    loo_emulator.fit(training_data, hyperparameters=gp.fit_hyperparameters)
    return loo_emulator.nes_error(left_out_datum.input, left_out_datum.output)


def pei(x: Input, gp: AbstractGaussianProcess) -> float:
    raise NotImplementedError


def compute_single_level_loo_samples(
    gp: AbstractGaussianProcess, domain: SimulatorDomain, batch_size: int = 1
) -> tuple[Input]:
    nes_loo_errors = tuple()  # TODO: fill in with LOO error calculations

    gp_e = None  # TODO: create GP from nes_loo_errors, probably of same type as type(gp) and with same hyperparameters(?)

    return maximise(lambda x: pei(x, gp_e), domain)
