import copy
from exauq.core.modelling import (
    TrainingDatum,
    AbstractEmulator,
    AbstractSimulator
)


class SingleLevelAdaptiveSampler:
    """A designer for training emulators using single level adaptive sampling.

    Implements the cross-validation-based adaptive sampling for emulators, as
    described in Mohammadi et. al. (2022).
    """
    def __init__(self, initial_design):
        self._initial_design = initial_design
    
    def __str__(self):
        return f"SingleLevelAdaptiveSampler designer with initial design {str(self._initial_design)}"

    def __repr__(self):
        return f"SingleLevelAdaptiveSampler(initial_design={repr(self._initial_design)})"
    
    def train(self, emulator: AbstractEmulator, simulator: AbstractSimulator):
        """Train an emulator with simulator outputs using this SLAS method."""
        return_emulator = copy.copy(emulator)
        initial_training_data = [TrainingDatum(x, simulator.compute(x))
                                 for x in self._initial_design]
        return_emulator.fit(initial_training_data)
        return return_emulator
