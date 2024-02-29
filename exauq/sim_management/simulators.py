import csv
import os
from abc import ABC, abstractmethod
from numbers import Real
from threading import Lock, Thread
from time import sleep, time
from typing import Any, Optional

from exauq.core.modelling import AbstractSimulator, Input, SimulatorDomain
from exauq.core.types import FilePath
from exauq.sim_management.hardware import HardwareInterface, JobStatus
from exauq.sim_management.jobs import Job, JobId
from exauq.utilities.csv_db import CsvDB, Record
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
    def _make_simulations_log(simulations_log: FilePath, input_dim: int):
        check_file_path(
            simulations_log,
            TypeError(
                "Argument 'simulations_log' must define a file path, but received "
                f"object of type {type(simulations_log)} instead."
            ),
        )

        return SimulationsLog(simulations_log, input_dim)

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
    input_dim : int
        The number of coordinates needed to define an input to the simultor.
    """

    def __init__(self, file: FilePath, input_dim: int):
        self._input_dim = self._validate_input_dim(input_dim)
        self._lock = Lock()
        self._job_id_key = "Job_ID"
        self._output_key = "Output"
        self._job_status_key = "Job_Status"
        self._input_keys = tuple(f"Input_{i}" for i in range(1, self._input_dim + 1))
        self._log_file_header = self._input_keys + (
            self._output_key,
            self._job_id_key,
            self._job_status_key,
        )
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
    def _validate_input_dim(input_dim: Any):
        """Check that the supplied arg is a positive integer, returning it if so."""

        if not isinstance(input_dim, int):
            raise TypeError(
                "Expected 'input_dim' to be of type integer, but received "
                f"{type(input_dim)} instead."
            )

        if not input_dim > 0:
            raise ValueError(
                "Expected 'input_dim' to be a positive integer, but received "
                f"{input_dim} instead."
            )

        return input_dim

    @staticmethod
    def _make_db(log_file: FilePath, fields: list[str]) -> CsvDB:
        """Make the underlying database used to store details of simulation jobs."""

        if isinstance(log_file, bytes):
            return CsvDB(log_file.decode(), fields)

        return CsvDB(log_file, fields)

    def _get_job_id(self, record: Record) -> str:
        """Get the job ID from a database record."""

        return record[self._job_id_key]

    def _get_output(self, record: Record) -> str:
        """Get the simulator output from a database record, as a string."""
        return record[self._output_key]

    def _get_job_status(self, record: Record) -> str:
        """Get the status of a job from a database record"""
        return record[self._job_status_key]

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

        return tuple(
            self._extract_simulation(record) for record in self._simulations_db.query()
        )

    def _extract_simulation(self, record: dict[str, str]) -> Simulation:
        """Extract a pair of simulator inputs and outputs from a dictionary record read
        from the log file. Missing outputs are converted to ``None``."""

        input_items = sorted(
            ((k, v) for k, v in record.items() if k.startswith("Input")),
            key=lambda x: x[0],
        )
        input_coords = (float(v) for _, v in input_items)
        x = Input(*input_coords)
        output = self._get_output(record)
        y = float(output) if output else None
        return x, y

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
        if len(x) != self._input_dim:
            raise ValueError(
                f"Expected input 'x' to have {self._input_dim} coordinates, but got "
                f"{len(x)} instead."
            )

        record = {h: "" for h in self._log_file_header}
        record.update(dict(zip(self._input_keys, x)))
        if job_id is not None:
            record.update({self._job_id_key: job_id})

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

        matched_records = self._simulations_db.query(
            lambda x: self._get_job_id(x) == job_id
        )
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

        new_record = matched_records[0] | {self._output_key: result}
        self._simulations_db.update(self._job_id_key, job_id, new_record)

    def get_pending_jobs(self) -> tuple[str]:
        """Return the IDs of all submitted jobs which don't have results.

        A job is considered to have been submitted if the corresponding record in the
        simulations log contains a job ID.

        Returns
        -------
        tuple[str]
            The IDs of all jobs that have been submitted but don't have a result recorded.
        """

        pending_records = self._simulations_db.query(
            lambda x: self._get_job_id(x) != "" and self._get_output(x) == ""
        )
        return tuple(self._get_job_id(record) for record in pending_records)

    def get_unsubmitted_inputs(self) -> tuple[Input]:
        """Get all simulator inputs that have not been submitted as jobs.

        This is defined to be the collection of inputs in the log file that do not have a
        corresponding job ID.

        Returns
        -------
        tuple[Input]
            The inputs that have not been submitted as jobs.
        """

        unsubmitted_records = self._simulations_db.query(
            lambda x: self._get_job_id(x) == ""
        )
        return tuple(
            self._extract_simulation(record)[0] for record in unsubmitted_records
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
        self._lock = Lock()
        self._thread = None

        self._id_generator = JobIDGenerator()

        self._job_strategies = self._init_job_strategies()

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

        job = Job(self._id_generator.generate_id(), x)
        try:
            self._interface.submit_job(job)
        finally:
            self._simulations_log.add_new_record(x, str(job.id))

        self._monitor([job])

    @staticmethod
    def _init_job_strategies() -> dict:
        strategies = {
            JobStatus.COMPLETED: CompletedJobStrategy(),
            JobStatus.FAILED: FailedJobStrategy(),
            JobStatus.RUNNING: RunningJobStrategy(),
            JobStatus.SUBMITTED: SubmittedJobStrategy(),
            JobStatus.NOT_SUBMITTED: NotSubmittedJobStrategy(),
            JobStatus.CANCELLED: CancelledJobStrategy(),
        }

        return strategies

    def _monitor(self, jobs: list[Job]):
        """Start monitoring the given job IDs."""

        with self._lock:
            self._jobs.extend(jobs)
        if self._thread is None or not self._thread.is_alive():
            self._thread = Thread(target=self._monitor_jobs)
            self._thread.start()

    def _monitor_jobs(self):
        """Continuously monitor the status of jobs and handle their completion."""

        while self._jobs:
            with self._lock:
                jobs = self._jobs[:]
            for job in jobs:
                status = self._interface.get_job_status(job.id)
                self.handle_job(job, status)
            if self._jobs:
                sleep(self._polling_interval)

    @property
    def interface(self):
        return self._interface

    @property
    def simulations_log(self):
        return self._simulations_log

    def remove_job(self, job: Job):
        with self._lock:
            self._jobs.remove(job)

    def handle_job(self, job: Job, status: JobStatus):
        if status:
            strategy = self._job_strategies.get(status)
            if strategy:
                strategy.handle(job, self)


class JobStrategy(ABC):
    @abstractmethod
    def handle(self, job: Job, job_manager: JobManager):
        raise NotImplementedError


class CompletedJobStrategy(JobStrategy):
    def handle(self, job: Job, job_manager: JobManager):
        result = job_manager.interface.get_job_output(job.id)
        job_manager.simulations_log.insert_result(str(job.id), result)
        job_manager.remove_job(job)


class FailedJobStrategy(JobStrategy):
    def handle(self, job: Job, job_manager: JobManager):
        pass


class RunningJobStrategy(JobStrategy):
    def handle(self, job: Job, job_manager: JobManager):
        pass


class SubmittedJobStrategy(JobStrategy):
    def handle(self, job: Job, job_manager: JobManager):
        pass


class NotSubmittedJobStrategy(JobStrategy):
    def handle(self, job: Job, job_manager: JobManager):
        pass


class CancelledJobStrategy(JobStrategy):
    def handle(self, job: Job, job_manager: JobManager):
        pass


class JobIDGenerator:
    """
    A generator for unique job IDs, encapsulated within a JobId object, based on the
    current time in milliseconds and a counter.

    This class ensures thread-safe generation of unique job IDs by combining the
    current timestamp in milliseconds with a counter that increments for each ID
    generated within the same millisecond. If the timestamp changes, the counter
    is reset. The uniqueness of each ID is maintained even in high concurrency
    scenarios.

    Methods
    -------
    generate_id() -> JobId
        Generates a unique JobId object representing the job ID.

    Examples
    --------
    >>> id_generator = JobIDGenerator()
    >>> job_id = id_generator.generate_id()
    >>> print(job_id)
    JobId('1589409675023001')
    """

    def __init__(self):
        """
        Initialises the JobIDGenerator with a new lock, sets the counter to 0, and
        the last timestamp to None.
        """
        self.lock = Lock()
        self.counter = 0
        self.last_timestamp = None

    def generate_id(self) -> 'JobId':
        """
        Generates a unique job ID encapsulated within a JobId object, based on the
        current timestamp and an internal counter.

        This method is thread-safe. It generates IDs by concatenating the current
        timestamp in milliseconds with a counter value. The counter increments with
        each call within the same millisecond and resets when the timestamp changes,
        ensuring each generated ID is unique.

        Returns
        -------
        JobId
            A JobId object encapsulating a unique job ID, consisting only of digits.
            The ID combines the current timestamp in milliseconds with a counter
            to ensure uniqueness.

        Examples
        --------
        >>> id_generator = JobIDGenerator()
        >>> job_id = id_generator.generate_id()
        >>> print(job_id)
        JobId('1589409675023001')
        """
        with self.lock:
            timestamp = int(time() * 1000)

            if timestamp == self.last_timestamp:
                self.counter += 1
            else:
                self.counter = 0
                self.last_timestamp = timestamp

            unique_id = f"{timestamp}{self.counter:03d}"

            return JobId(unique_id)

