import copy
from collections.abc import Collection
from exauq.core.modelling import Input, TrainingDatum, AbstractEmulator


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

    def new_design_batch(self, size: int = 1):

        return [Input(1)] * size
