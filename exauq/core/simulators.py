from numbers import Real
from typing import Optional

from exauq.core.modelling import AbstractSimulator, Input


class Simulator(AbstractSimulator):
    def compute(self, x: Input) -> Optional[Real]:
        """
        Compute the simulation output of an input to the simulator.
        
        Parameters
        ----------
        x : Input
            An input for the simulator.

        Returns
        -------
        Optional[Real]
            ``None`` if a new input has been provided.
        """
        pass
