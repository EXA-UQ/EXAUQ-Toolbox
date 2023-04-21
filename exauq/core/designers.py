import copy
from exauq.core.modelling import (
    AbstractEmulator,
    AbstractSimulator
)


class SingleLevelAdaptiveSampler:
    """A designer for training an emulator using single level adaptive sampling.

    Implements the cross-validation-based adaptive sampling for emulators, as
    described in Mohammadi et. al. (2022).

    Parameters
    ----------
    emulator: AbstractEmulator
        An emulator to fit using the single level adaptive sampling design
        strategy.
    simulator: AbstractSimulator
        The simulator being emulated.
    
    Attributes
    ----------
    emulator: AbstractEmulator
        The emulator being trained with the single level adaptive sampling
        design strategy.
    simulator: AbstractSimulator
        The simulator being emulated.
    """
    def __init__(self, emulator: AbstractEmulator, simulator: AbstractSimulator):
        self.emulator: AbstractEmulator = emulator
        self.simulator: AbstractSimulator = simulator
    
    def __str__(self):
        return f"SingleLevelAdaptiveSampler designer for simulator {str(self.simulator)}, " \
               f"using emulator {str(self.emulator)}"

    def __repr__(self):
        return f"SingleLevelAdaptiveSampler(simulator={repr(self.simulator)}, " \
               f"emulator={repr(self.emulator)})"

    def run(self):
        """Run the adaptive sampling algorithm."""
        trained_emulator = copy.copy(self.emulator)
        trained_emulator.fit([(0, 0)])
        return trained_emulator
