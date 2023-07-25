from numbers import Real
from os import PathLike
from typing import Optional, Union

from exauq.core.modelling import AbstractSimulator, Input


class Simulator(AbstractSimulator):
    """
    Represents a simulation code that can be run on inputs.
    
    Parameters
    ----------
    simulations_log : str or bytes or os.PathLike
        A path to the simulation log file.
    """
    def __init__(self, simulations_log: Union[str, bytes, PathLike]):
        pass

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
