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
        The emulator being trained with the single levels adaptive sampling
        design strategy.
    simulator: AbstractSimulator
        The simulator being emulated.
    """
    def __init__(self, emulator: AbstractEmulator, simulator: AbstractSimulator):
        self.emulator: AbstractEmulator = emulator
        self.simulator: AbstractSimulator = simulator
    
    def __str__(self):
        return f"SingleLevelAdaptiveSampling designer for simulator {str(self.simulator)}, " \
               f"using emulator {str(self.emulator)}"

    def __repr__(self):
        return f"SingleLevelAdaptiveSampling(simulator={repr(self.simulator)}, " \
               f"emulator={repr(self.emulator)})"
