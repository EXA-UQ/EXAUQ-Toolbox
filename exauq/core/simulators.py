import csv
import os

from numbers import Real
from threading import Thread, Lock
from time import sleep
from typing import Optional

from exauq.core.hardware import HardwareInterface
from exauq.core.modelling import AbstractSimulator, Input, SimulatorDomain
from exauq.core.types import FilePath
from exauq.utilities.validation import check_file_path


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
        (Default: ``simulations.csv``) A path to the simulation log file. The default
        will work with a file called ``simulations.csv`` in the current working directory
        for the calling Python process.

    Attributes
    ----------
    previous_simulations : tuple
        (Read-only) Simulations that have been previously submitted for evaluation.
    """

    # TODO: rename simulations_log
    def __init__(self, domain: SimulatorDomain, interface: HardwareInterface, simulations_log: FilePath = "simulations.csv"):
        self._simulations_log = self._make_simulations_log(simulations_log, domain.dim)
        self._manager = JobManager(self._simulations_log, interface)
        self._previous_simulations = list(self._simulations_log.get_simulations())

    @staticmethod
    def _make_simulations_log(simulations_log: FilePath, num_inputs: int):
        check_file_path(
            simulations_log,
            TypeError(
                "Argument 'simulations_log' must define a file path, but received "
                f"object of type {type(simulations_log)} instead."
            ),
        )

        return SimulationsLog(simulations_log, num_inputs)

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
            ``None`` if a new input has been provided, or else the corresponding simulator
            output, if this has previously been computed.
        """

        if not isinstance(x, Input):
            raise TypeError(
                f"Argument 'x' must be of type Input, but received {type(x)}."
            )

        for _input, output in self._previous_simulations:
            if _input == x:
                return output

        self._manager.submit(x)
        self._previous_simulations.append((x, None))
        return None


class SimulationsLog(object):
    """
    An interface to a log file containing details of simulations.

    The log file is a csv file containing a record of simulations that have been submitted
    for computation; it will be created at the supplied file path upon initialisation. The
    input of each submission is recorded along with the simulator output, if this has been
    computed. Columns that give the input coordinates should have headings 'Input_n' where
    ``n`` is the index of the coordinate (starting at 1). The column giving the simulator
    output should have the heading 'Output'.

    Parameters
    ----------
    file : str, bytes or path-like
        A path to the underlying log file containing details of simulations.
    """

    def __init__(self, file: FilePath, num_inputs: int):
        self._log_file = self._initialise_log_file(file, num_inputs)

    @staticmethod
    def _initialise_log_file(file: FilePath, num_inputs: int) -> FilePath:
        """Create a new file at the given path if it doesn't already exist and return
        the path."""

        check_file_path(
            file,
            TypeError(
                "Argument 'file' must define a file path, but received object of "
                f"type {type(file)} instead."
            ),
        )

        if not os.path.exists(file):
            with open(file, mode="w", newline="") as _file:
                writer = csv.writer(_file)
                header = [f"Input_{i + 1}" for i in range(num_inputs)] + ["Output", "Job_ID"]
                writer.writerow(header)

        return file

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
        inputs and outputs. Missing outputs are converted to ``None``."""

        input_items = sorted(
            ((k, v) for k, v in record.items() if k.startswith("Input")),
            key=lambda x: x[0],
        )
        input_coords = (float(v) for _, v in input_items)
        x = Input(*input_coords)
        y = float(record["Output"]) if record["Output"] else None
        return x, y

    def add_new_record(self, x: Input):
        with open(self._log_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(list(x) + [""])

    def insert_job_id(self, input_set: Input, job_id):
        with open(self._log_file, 'r') as csvfile:
            reader = csv.reader(csvfile)
            rows = list(reader)

        header = rows[0]
        data_rows = rows[1:]

        job_id_index = header.index('Job_ID')

        for i, row in enumerate(data_rows):
            if row[:len(input_set)] == input_set:
                row[job_id_index] = job_id

        with open(self._log_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            writer.writerows(data_rows)

    def insert_result(self, job_id, result):
        with open(self._log_file, 'r') as csvfile:
            reader = csv.reader(csvfile)
            rows = list(reader)

        header = rows[0]
        data_rows = rows[1:]

        result_index = header.index('Output')
        job_id_index = header.index('Job_ID')

        for i, row in enumerate(data_rows):
            if row[job_id_index] == job_id:
                row[result_index] = result

        with open(self._log_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            writer.writerows(data_rows)


class JobManager(object):
    def __init__(self, simulations_log: SimulationsLog, interface: HardwareInterface):
        self._simulations_log = simulations_log
        self._interface = interface
        self._manager = Manager()
        self._jobs = self._manager.list()
        self._running = Value('b', False)

    def submit(self, x: Input):
        self._simulations_log.add_new_record(x)
        job_id = self._interface.submit_job(x)
        self._jobs.append(job_id)
        if not self._running.value:
            process = Process(target=self._monitor_jobs)
            process.start()

    def _monitor_jobs(self):
        self._running.value = True
        while self._jobs:
            for job_id in self._jobs:
                status = self._interface.get_job_status(job_id)
                if status: # Currently job status either true (job complete) or false (not complete)
                    result = self._interface.get_job_output(job_id)
                    self._simulations_log.insert_result(job_id, result)
                    self._jobs.remove(job_id)
            sleep(1)  # Add param...
        self._running.value = False
