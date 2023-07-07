import copy

from exauq.core.modelling import (
    AbstractEmulator,
    AbstractSimulator,
    Input,
    TrainingDatum,
)
from exauq.utilities.validation import check_int


class RandomSamplerDesigner:
    def new_design_points(self, size: int):
        check_int(size, TypeError("Argument 'size' must be of type 'int'."))
        return [None] * size


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
