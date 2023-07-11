import copy

import numpy as np

from exauq.core.modelling import (
    AbstractEmulator,
    AbstractSimulator,
    Input,
    SimulatorDomain,
    TrainingDatum,
)
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

    def new_design_points(self, size: int) -> list[Input]:
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
            size, TypeError(f"Expected 'size' of type 'int' but received {type(size)}.")
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
    initial_design: list[Experiment]
        A list of design points to form the basis of an initial training
        dataset.
    """

    def __init__(self, initial_design: list[Input]):
        self._initial_design = initial_design

    def __str__(self) -> str:
        return f"SingleLevelAdaptiveSampler designer with initial design {str(self._initial_design)}"

    def __repr__(self) -> str:
        return (
            f"SingleLevelAdaptiveSampler(initial_design={repr(self._initial_design)})"
        )

    def train(
        self, emulator: AbstractEmulator, simulator: AbstractSimulator
    ) -> AbstractEmulator:
        """Train an emulator with simulator outputs using this SLAS method.

        Parameters
        ----------
        emulator : AbstractEmulator
            The emulator to train.
        simulator : AbstractSimulator
            The simulator to be emulated.

        Returns
        -------
        AbstractEmulator
            A new emulator that has been trained with observations produced by
            the given simulator, using the SLAS methodology. A new object is
            returned of the same ``type`` as `emulator`.
        """
        return_emulator = copy.copy(emulator)
        initial_training_data = [
            TrainingDatum(x, simulator.compute(x)) for x in self._initial_design
        ]
        return_emulator.fit(initial_training_data)
        return return_emulator
