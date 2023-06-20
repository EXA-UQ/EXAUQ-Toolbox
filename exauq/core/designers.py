import copy
from exauq.core.modelling import (
    Input,
    TrainingDatum,
    AbstractEmulator,
    AbstractSimulator
)


class SingleLevelAdaptiveSampler:
    """Single level adaptive sampling (SLAS) for training emulators.

    Implements the cross-validation-based adaptive sampling for emulators, as
    described in Mohammadi et. al. (2022).

    Parameters
    ----------
    initial_data: list[TrainingDatum]
        Training data on which the emulator will initially be trained.
    """
    def __init__(self, initial_data: list[TrainingDatum]):
        self._initial_data = self._validate_initial_data(initial_data)

    @staticmethod
    def _validate_initial_data(initial_data):
        if not initial_data:
            raise ValueError(
                "SingleLevelAdaptiveSampler must be initialised with nonempty training "
                "data"
            )

        return initial_data

    def __str__(self) -> str:
        return f"SingleLevelAdaptiveSampler designer with initial data {str(self._initial_data)}"

    def __repr__(self) -> str:
        return f"SingleLevelAdaptiveSampler(initial_data={repr(self._initial_data)})"
    
    def train(self, emulator: AbstractEmulator, simulator: AbstractSimulator) -> AbstractEmulator:
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
        return_emulator.fit(self._initial_data)
        return return_emulator
