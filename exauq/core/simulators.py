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
    simulation outputs that have been recorded in a simulations log file, if supplied.
    In the case where an `Input` has been submitted for evaluation but no output from
    the simulator has been retrieved, the output is recorded as ``None``.

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
        if simulations_log is None:
            self._previous_simulations = []
        else:
            self._previous_simulations = list(
                SimulationsLog(simulations_log).get_simulations()
            )

    @property
    def previous_simulations(self) -> tuple:
        """
        (Read-only) A tuple of simulations that have been previously submitted for
        computation.
        """
        return tuple(self._previous_simulations)

    def compute(self, x: Input) -> Optional[Real]:
        """
        Submit a simulation input for computation.

        In the case where a never-before seen input is supplied, this will be submitted
        for computation and ``None`` will be returned. In the case where an input has been
        seen before (that is, features in an entry in the simulations log file for this
        simulator), the corresponding simulator output will be returned, if this is
        available.

        Parameters
        ----------
        x : Input
            An input for the simulator.

        Returns
        -------
        Optional[Real]
            ``None`` if a new input has been provided or corresponding simulator output,
            if this has previously been computed.
        """

        for _input, output in self._previous_simulations:
            if _input == x:
                return output

        self._previous_simulations.append((x, None))
        return None


class SimulationsLog(object):
    """
    An interface to a log file containing details of simulations.

    The log file contains a record of simulations that have been submitted for
    computation. The input of each submission is recorded along with the simulator
    output, if this has been computed. This is recorded in csv format. Columns that give
    the input coordinates should have headings 'Input_n' where ``n`` is the index of
    the coordinate (starting at 1). The column giving the simulator output should have
    the heading 'Output'.

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

        with open(self._log_file, mode="r", newline="") as log_file:
            return tuple(map(self._parse_row, csv.DictReader(log_file)))

    @staticmethod
    def _parse_row(record: dict[str, str]) -> tuple[Input, Optional[Real]]:
        """Convert a dictionary record read from the log file into a pair of simulator
        inputs and outputs. Missing outputs are converted to the empty string."""

        input_items = sorted(
            ((k, v) for k, v in record.items() if k.startswith("Input")),
            key=lambda x: x[0],
        )
        input_coords = (float(v) for _, v in input_items)
        x = Input(*input_coords)
        y = float(record["Output"]) if record["Output"] else None
        return x, y
