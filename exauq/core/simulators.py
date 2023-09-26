import csv
import os
from numbers import Real
from threading import Lock, Thread
from time import sleep
from typing import Any, Optional

from exauq.core.hardware import HardwareInterface
from exauq.core.modelling import AbstractSimulator, Input, SimulatorDomain
from exauq.core.types import FilePath
from exauq.utilities.csv_db import CsvDB
from exauq.utilities.validation import check_file_path

Simulation = tuple[Input, Optional[Real]]
"""A type to represent a simulator input, possibly with corresponding simulator output."""


class Simulator(AbstractSimulator):
    """
    Represents a simulation code that can be run with inputs.

    This class provides a way of working with some simulation code as a black box,
    abstracting away details of how to submit simulator inputs for computation and
    retrieve the results. The set of valid simulator inputs is represented by the
    `domain` supplied at initialisation, while the `interface` provided at initialisation
    encapsulates the details of how to actually run the simulation code with a
    collection of inputs and retrieve outputs.

    A simulations log file records inputs that have been submitted for computation, as
    well as any simulation outputs. These can be retrieved using the
    ``previous_simulations` property of this class.

    Parameters
    ----------
    domain : SimulatorDomain
        The domain of the simulator, representing the set of valid inputs for the
        simulator.
    interface : HardwareInterface
        An implementation of the ``HardwareInterface`` base class, providing the interface
        to a computer that the simulation code runs on.
    simulations_log_file : str or bytes or os.PathLike, optional
        (Default: ``simulations.csv``) A path to the simulation log file. The default
        will work with a file called ``simulations.csv`` in the current working directory
        for the calling Python process.

    Attributes
    ----------
    previous_simulations : tuple[Input, Optional[Real]]
        (Read-only) Simulations that have been previously submitted for evaluation.
        In the case where an `Input` has been submitted for evaluation but no output from
        the simulator has been retrieved, the output is recorded as ``None``.
    """

    def __init__(
        self,
        domain: SimulatorDomain,
        interface: HardwareInterface,
        simulations_log_file: FilePath = "simulations.csv",
    ):
        self._check_arg_types(domain, interface)
        self._simulations_log = self._make_simulations_log(
            simulations_log_file, domain.dim
        )
        self._manager = JobManager(self._simulations_log, interface)

    @staticmethod
    def _check_arg_types(domain: Any, interface: Any):
        if not isinstance(domain, SimulatorDomain):
            raise TypeError(
                "Argument 'domain' must define a SimulatorDomain, but received object "
                f"of type {type(domain)} instead."
            )

        if not isinstance(interface, HardwareInterface):
            raise TypeError(
                "Argument 'interface' must inherit from HardwareInterface, but received "
                f"object of type {type(interface)} instead."
            )

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
    def previous_simulations(self) -> Simulation:
        """
        (Read-only) A tuple of simulations that have been previously submitted for
        computation. In the case where an `Input` has been submitted for evaluation but
        no output from the simulator has been retrieved, the output is recorded as
        ``None``.
        """
        return tuple(self._simulations_log.get_simulations())

    def compute(self, x: Input) -> Optional[Real]:
        """
        Submit a simulation input for computation.

        In the case where a never-before seen input is supplied, this will be submitted
        for computation and ``None`` will be returned. If the input has been seen before
        (that is, features in an entry in the simulations log file for this simulator),
        then the corresponding simulator output will be returned, or ``None`` if this is
        not yet available.

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

        for _input, output in self.previous_simulations:
            if _input == x:
                return output

        self._manager.submit(x)
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
    num_inputs : int
        The number of coordinates needed to define an input to the simultor.
    """

    def __init__(self, file: FilePath, num_inputs: int):
        self._log_file_header = [f"Input_{i}" for i in range(1, num_inputs + 1)] + [
            "Output",
            "Job_ID",
        ]
        self._log_file = self._initialise_log_file(file)
        self._simulations_db = self._make_db(self._log_file, self._log_file_header)

    def _initialise_log_file(self, file: FilePath) -> FilePath:
        """Create a new simulations log file at the given path if it doesn't already exist
        and return the path."""

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
                writer.writerow(self._log_file_header)

        return file

    @staticmethod
    def _make_db(log_file: FilePath, fields: list[str]) -> CsvDB:
        if isinstance(log_file, bytes):
            return CsvDB(log_file.decode(), fields)

        return CsvDB(log_file, fields)

    def get_simulations(self) -> tuple[Simulation]:
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

        return tuple(map(self._extract_simulation, self.get_records()))

    @staticmethod
    def _extract_simulation(record: dict[str, str]) -> Simulation:
        """Extract a pair of simulator inputs and outputs from a dictionary record read
        from the log file. Missing outputs are converted to ``None``."""

        input_items = sorted(
            ((k, v) for k, v in record.items() if k.startswith("Input")),
            key=lambda x: x[0],
        )
        input_coords = (float(v) for _, v in input_items)
        x = Input(*input_coords)
        y = float(record["Output"]) if record["Output"] else None
        return x, y

    def get_records(self, job_ids: Optional[set[str]] = None) -> tuple[dict[str, str]]:
        """Retrieve records of jobs from the simulations log file.

        Records are represented as dictionaries where the keys are the headings in the
        log file and values are string representations of the corresponding values in the
        log file.

        Parameters
        ----------
        job_ids : set[str] or None, optional
            (Default: ``None``) The IDs of jobs whose records should be retrieved. If
            ``None`` then records for all jobs will be retrieved.

        Returns
        -------
        tuple[dict[str, str]]
            The log file records for the supplied job IDs.
        """

        if job_ids is None:
            return self._simulations_db.query()

        return self._simulations_db.query(
            lambda x: x[self._log_file_header[-1]] in job_ids
        )

    def create_record(self, record: dict[str, Any]) -> None:
        """Creates a new record in the simulations log file.

        Parameters
        ----------
        record : dict[str, Any]
            The record to write to the simulations log file, as a dictionary mapping
            log file column headings to the corresponding values.

        Raises
        ------
        ValueError
            If the keys of the supplied `record` don't correspond to the log file
            headings.
        """

        self._check_fields(record)
        with open(self._log_file, mode="a", newline="") as csvfile:
            csv.DictWriter(csvfile, fieldnames=self._log_file_header).writerow(record)

    def _check_fields(self, record: dict[str, Any]):
        """Check that a record has exactly the fields required for the simlations log
        file."""

        fields_missing = [h for h in self._log_file_header if h not in record]
        if fields_missing:
            raise ValueError(
                "The record does not contain entries for the required fields: "
                f"{', '.join(fields_missing)}."
            )

        extra_fields = [h for h in sorted(record) if h not in self._log_file_header]
        if extra_fields:
            raise ValueError(
                "The record contains fields not in the simulations log file: "
                f"{', '.join(extra_fields)}."
            )

    def add_new_record(self, x: Input, job_id: Optional[str] = None) -> None:
        """Record a new simulation job in the log file.

        Parameters
        ----------
        x : Input
            An input for the simulator to evaluate.
        job_id: str, Optional
            (Default: ``None``) The ID for the job of evaluating the simulator at `x`.
            If ``None`` then no job ID will be recorded alongside the input `x` in the
            simulations log file.
        """
        record = {h: "" for h in self._log_file_header}
        record.update(
            dict(zip([h for h in self._log_file_header if h.startswith("Input")], x))
        )
        if job_id is not None:
            record.update({"Job_ID": job_id})

        self._simulations_db.create(record)

    def insert_result(self, job_id: str, result: Real) -> None:
        """Insert the output of a simulation into a job record in the simulations log
        file.

        Parameters
        ----------
        job_id : str
            The ID of the job that the `result` should be added to.
        result : Real
            The output of a simulation.

        Raises
        ------
        SimulationsLogLookupError
            If there isn't a unique simulations log record having job ID `job_id`.
        """

        job_id_key = "Job_ID"
        matched_records = self._simulations_db.query(lambda x: x[job_id_key] == job_id)
        num_matched_records = len(matched_records)

        if num_matched_records != 1:
            msg = (
                (
                    f"Could not add output to simulation with job ID = {job_id}: "
                    "no such simulation exists."
                )
                if num_matched_records == 0
                else (
                    f"Could not add output to simulation with job ID = {job_id}: "
                    "multiple records with this ID found."
                )
            )
            raise SimulationsLogLookupError(msg)

        new_record = matched_records[0] | {"Output": result}
        self._simulations_db.update(job_id_key, job_id, new_record)

    def get_pending_jobs(self) -> tuple[str]:
        """Return the IDs of all submitted jobs which don't have results.

        A job is considered to have been submitted if the corresponding record in the
        simulations log contains a job ID.

        Returns
        -------
        tuple[str]
            The IDs of all jobs that have been submitted but don't have a result recorded.
        """
        return tuple(
            record["Job_ID"]
            for record in self.get_records()
            if record["Job_ID"] and not record["Output"]
        )


class SimulationsLogLookupError(Exception):
    """Raised when a simulations log does not contain a particular record."""

    pass


class JobManager(object):
    """
    Manages and monitors simulation jobs.

    The `JobManager` class handles the submission and monitoring of simulation jobs
    through the given hardware interface. It ensures that the results of completed
    jobs are logged and provides functionality to handle pending jobs.

    Parameters
    ----------
    simulations_log : SimulationsLog
        The log object where simulations are recorded.
    interface : HardwareInterface
        The hardware interface to interact with the simulation hardware.
    polling_interval : int, optional
        The interval in seconds at which the job status is polled. Default is 10.
    wait_for_pending : bool, optional
        If True, waits for pending jobs to complete during initialization. Default is True.
    """

    def __init__(
        self,
        simulations_log: SimulationsLog,
        interface: HardwareInterface,
        polling_interval: int = 10,
        wait_for_pending: bool = True,
    ):
        self._simulations_log = simulations_log
        self._interface = interface
        self._polling_interval = polling_interval
        self._jobs = []
        self._running = False
        self._lock = Lock()
        self._thread = None

        self._monitor(self._simulations_log.get_pending_jobs())
        if wait_for_pending and self._thread is not None:
            self._thread.join()

    def submit(self, x: Input):
        """Submit a new simulation job.

        If the job gets submitted without error, then the simulations log file will
        have a record of the corresponding simulator input along with a job ID.
        Conversely, if there is an error in submitting the job then only the input
        is recorded in the log file, with blank job ID.
        """

        job_id = None
        try:
            job_id = self._interface.submit_job(x)
        finally:
            self._simulations_log.add_new_record(x, job_id)

        self._monitor([job_id])

    def _monitor(self, job_ids: list):
        """Start monitoring the given job IDs."""

        with self._lock:
            self._jobs.extend(job_ids)
        if not self._running and self._jobs:
            self._thread = Thread(target=self._monitor_jobs)
            self._thread.start()

    def _monitor_jobs(self):
        """Continuously monitor the status of jobs and handle their completion."""

        with self._lock:
            self._running = True
        while self._jobs:
            with self._lock:
                job_ids = self._jobs[:]
            for job_id in job_ids:
                status = self._interface.get_job_status(job_id)
                if status:  # Job complete
                    result = self._interface.get_job_output(job_id)
                    self._simulations_log.insert_result(job_id, result)
                    with self._lock:
                        self._jobs.remove(job_id)
            if self._jobs:
                sleep(self._polling_interval)
        with self._lock:
            self._running = False
