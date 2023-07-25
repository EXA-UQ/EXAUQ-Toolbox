from numbers import Real
from os import PathLike
from typing import Optional, Union

from exauq.core.modelling import AbstractSimulator, Input


class Simulator(AbstractSimulator):
    """
    Represents a simulation code that can be run on inputs.

    Simulations that have been previously submitted for computation can be retrieved
    using the ``previous_simulations` property. This returns a tuple of `Input`s and
    simulation outputs; in the case where an `Input` has been submitted for evaluation
    but no output from the simulator has been retrieved, the output is recorded as
    ``None``.

    Parameters
    ----------
    simulations_log : str or bytes or os.PathLike, optional
        (Default: ``None``) A path to the simulation log file.

    Attributes
    ----------
    previous_simulations : tuple
        (Read-only) Simulations that have been previously submitted for evaluation.
    """
    def __init__(self, simulations_log: Optional[Union[str, bytes, PathLike]] = None):
        self._previous_simulations = tuple()

    @property
    def previous_simulations(self) -> tuple:
        """
        (Read-only) A tuple of simulations that have been previously submitted for
        computation.
        """
        return self._previous_simulations

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
