import csv
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


class SimulationsLog(object):
    """
    An interface to a log file containing details of simulations.

    The log file contains a record of simulations that have been submitted for
    computation. The input of each submission is recorded along with the simulator
    output, if this has been computed. This is recorded in csv format.

    Parameters
    ----------
    file : str, bytes or path-like, optional
        A path to the underlying log file containing details of simulations.
    """
    def __init__(self, file: Optional[Union[str, bytes, PathLike]] = None):
        self._log_file = file

    def get_simulations(self):
        """
        Get all simulations contained in the log file.

        This returns an immutable sequence of simulator inputs along with outputs. In
        the case where the simulator output is not available for the corresponding
        input, ``None`` is instead returned alongside the input.

        Returns
        -------
        tuple[tuple[Input, Optional[Real]]]
            A tuple of ``(x, y)`` pairs, where ``x`` is an `Input` and ``y`` is the
            simulation output, or ``None`` if this hasn't yet been computed.
        """
        simulations = []
        with open(self._log_file, mode="r", newline="") as log_file:
            for row in csv.DictReader(log_file):
                x = Input(float(row["Input"]))
                y = float(row["Output"]) if row["Output"] else None
                simulations.append((x, y))

        return tuple(simulations)
