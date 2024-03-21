import csv
import os
import random
from abc import ABC, abstractmethod
from datetime import datetime
from numbers import Real
from threading import Lock, Thread
from time import sleep
from typing import Any, Optional, Union

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
    def previous_simulations(self) -> tuple[Simulation]:
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
    def _make_db(log_file: FilePath, fields: tuple[str]) -> CsvDB:
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

    def _get_job_status(self, record: Record) -> JobStatus:
        """Get the status of a job from a database record"""
        status_str = record[self._job_status_key]
        return JobStatus(status_str)

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
        with self._lock:
            return tuple(
                self._extract_simulation(record)
                for record in self._simulations_db.query()
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

    def add_new_record(
        self,
        x: Input,
        job_id: Union[str, JobId, int],
        job_status: JobStatus = JobStatus.NOT_SUBMITTED,
    ) -> None:
        """
        Record a new simulation job in the log file.

        This method adds a new record for a simulation job with a given input,
        job ID, and job status. It ensures that the job ID is unique and not None,
        and that the input dimension matches the expected dimension.

        Parameters
        ----------
        x : Input
            An input for the simulator to evaluate.
        job_id : Union[str, JobId, int]
            The ID for the job of evaluating the simulator at `x`. Must consist only of digits and cannot be None.
        job_status : JobStatus, optional
            The status of the job to be recorded alongside the input `x`.
            Defaults to JobStatus.NOT_SUBMITTED.

        Raises
        ------
        ValueError
            - If `job_id` is None.
            - If the input `x` does not have the expected number of coordinates.
            - If the `job_id` is already in use.

        """
        with self._lock:
            if job_id is None:
                raise ValueError("job_id cannot be None.")

            if len(x) != self._input_dim:
                raise ValueError(
                    f"Expected input 'x' to have {self._input_dim} coordinates, but got "
                    f"{len(x)} instead."
                )

            existing_record = self._simulations_db.retrieve(self._job_id_key, str(job_id))
            if existing_record:
                raise ValueError(f"The job_id '{job_id}' is already in use.")

            record = {h: "" for h in self._log_file_header}
            record = {
                **record,
                **dict(zip(self._input_keys, x)),
                self._job_id_key: str(job_id),
                self._job_status_key: job_status.value,
            }

            self._simulations_db.create(record)

    def insert_result(self, job_id: Union[str, JobId], result: Real) -> None:
        """Insert the output of a simulation into a job record in the simulations log
        file.

        Parameters
        ----------
        job_id : Union[str, JobId]
            The ID of the job that the `result` should be added to.
        result : Real
            The output of a simulation.

        Raises
        ------
        SimulationsLogLookupError
            If there isn't a log record having job ID `job_id`.
        """

        job_id_str = str(job_id)

        with self._lock:
            record = self._simulations_db.retrieve(self._job_id_key, job_id_str)
            if record:
                new_record = record | {self._output_key: result}
                self._simulations_db.update(self._job_id_key, job_id_str, new_record)
            else:
                msg = (
                    f"Could not add output to simulation with job ID = {job_id_str}: "
                    "no such simulation exists."
                )
                raise SimulationsLogLookupError(msg)

    def get_pending_jobs(self) -> tuple[Job]:
        """Return all jobs which don't have results and are in non-terminal states.

        A job is considered pending if it is in one of the following states:
        RUNNING, SUBMITTED, NOT_SUBMITTED, or FAILED_SUBMIT.

        Returns
        -------
        tuple[Job]
            The Jobs that are in a pending state.
        """
        with self._lock:
            pending_statuses = {
                JobStatus.RUNNING,
                JobStatus.SUBMITTED,
                JobStatus.NOT_SUBMITTED,
                JobStatus.FAILED_SUBMIT,
            }
            pending_records = self._simulations_db.query(
                lambda x: self._get_job_status(x) in pending_statuses
            )
            return tuple(
                Job(self._get_job_id(record), self._extract_simulation(record)[0])
                for record in pending_records
            )

    def get_unsubmitted_inputs(self) -> tuple[Input]:
        """Get all simulator inputs that have not been submitted as jobs.

        Identifies inputs that are marked as 'NOT_SUBMITTED' in the simulation database,
        signaling they have not been dispatched for execution.

        Returns
        -------
        tuple[Input]
            The inputs that have not been submitted as jobs.
        """
        with self._lock:
            unsubmitted_records = self._simulations_db.query(
                lambda x: self._get_job_status(x) == JobStatus.NOT_SUBMITTED
            )
            return tuple(
                self._extract_simulation(record)[0] for record in unsubmitted_records
            )

    def update_job_status(self, job_id: Union[str, JobId], new_status: JobStatus) -> None:
        """
        Updates the status of a job in the simulations log.

        This method updates the job status for a given job ID in the simulation log database.
        It ensures thread safety by locking the operation. If the job ID does not exist it
        raises a `SimulationsLogLookupError`.

        Parameters
        ----------
        job_id : Union[str, JobId]
            The unique identifier of the job whose status is to be updated.
        new_status : JobStatus
            The new status to be assigned to the job. This must be an instance of the `JobStatus` enum.

        Raises
        ------
        SimulationsLogLookupError
            If there isn't a log record having job ID `job_id`.

        Examples
        --------
        Suppose we have a job with ID '12345' that we want to mark as completed. We would call the method as follows:

        >>> update_job_status('12345', JobStatus.COMPLETED)

        If the job ID '12345' does not exist in the log, a `SimulationsLogLookupError` will be raised.

        Notes
        -----
        This method is thread-safe and can be called concurrently from multiple threads without causing
        data corruption or race conditions.
        """
        job_id_str = str(job_id)

        with self._lock:
            record = self._simulations_db.retrieve(self._job_id_key, job_id_str)
            if record:
                new_record = record | {self._job_status_key: new_status.value}
                self._simulations_db.update(self._job_id_key, job_id_str, new_record)
            else:
                msg = (
                    f"Could not update status of simulation with job ID = {job_id_str}: "
                    "no such simulation exists."
                )
                raise SimulationsLogLookupError(msg)

    def get_job_status(self, job_id: Union[str, JobId]) -> JobStatus:
        """
        Retrieves the current status of a specified job from the simulations log.

        This method queries the simulations log database for a job with the given ID and
        returns its current status. It is thread-safe, ensuring consistent reads even when
        accessed concurrently from multiple threads. If the job ID does not exist in the
        database it raises an exception.

        Parameters
        ----------
        job_id : Union[str, JobId]
            The unique identifier of the job whose status is to be retrieved.

        Returns
        -------
        JobStatus
            The current status of the job as an instance of the `JobStatus` enum.

        Raises
        ------
        SimulationsLogLookupError
            If there isn't a log record having job ID `job_id`.

        Examples
        --------
        >>> get_job_status('12345')
        JobStatus.RUNNING

        This example returns the `JobStatus.RUNNING` enum, indicating that the job with
        ID '12345' is currently running.

        Notes
        -----
        This method is particularly useful for monitoring the progress of jobs and
        handling them based on their current state. It enforces data integrity by
        ensuring that each job ID is unique and correctly mapped to a valid job status.
        """
        job_id_str = str(job_id)

        with self._lock:
            record = self._simulations_db.retrieve(self._job_id_key, job_id_str)
            if record:
                return self._get_job_status(record)
            else:
                msg = (
                    f"Could not retrieve status of simulation with job ID = {job_id_str}: "
                    "no such simulation exists."
                )
                raise SimulationsLogLookupError(msg)


class SimulationsLogLookupError(Exception):
    """Raised when a simulations log does not contain a particular record."""

    pass


class JobManager:
    """
    Orchestrates the submission, monitoring, and status management of simulation jobs
    within a simulation environment. Utilizes a specified hardware interface for job
    execution and interacts with a simulations log for recording job activities.

    This manager supports dynamic job status updates, retry strategies for submission
    failures, and employs a strategy pattern for handling different job statuses, making
    the system adaptable to various simulation requirements and hardware interfaces.

    Parameters
    ----------
    simulations_log : SimulationsLog
        A log for recording and retrieving details of simulation jobs.
    interface : HardwareInterface
        An abstract interface to the hardware or simulation environment where jobs
        are executed.
    polling_interval : int, optional
        Time interval, in seconds, for polling job statuses during monitoring. Defaults to 10 seconds.
    wait_for_pending : bool, optional
        Specifies whether the manager should wait for all pending jobs to reach a
        conclusive status (e.g., COMPLETED or FAILED) upon initialization. Defaults to True.

    Methods
    -------
    submit(x: Input)
        Submits a new job based on the provided simulation input. Handles initial job
        logging and status setting to NOT_SUBMITTED.
    monitor(jobs: list[Job])
        Initiates or continues monitoring of job statuses in a separate background thread.

    Raises
    ------
    SimulationsLogLookupError
        If operations on the simulations log encounter inconsistencies, such as
        missing records or duplicate job IDs.

    Examples
    --------
    >>> job_manager = JobManager(simulations_log, hardware_interface)
    >>> input_data = Input(0.0, 1.0)
    >>> job_manager.submit(input_data)

    The job manager will handle the submission, monitor the job's progress, and update
    its status accordingly in the simulations log.
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

        self.monitor(self._simulations_log.get_pending_jobs())
        if wait_for_pending and self._thread is not None:
            self._thread.join()

    def submit(self, x: Input):
        """
        Initialises a new simulation job, logs it with a NOT_SUBMITTED status, and then
        passes it to the job handling strategies for submission.

        This method generates a unique ID for the job and records it in the simulations log.
        The job, marked as NOT_SUBMITTED, is then handed over to the appropriate job handling
        strategy, which is responsible for submitting the job to the simulation hardware.

        Parameters
        ----------
        x : Input
            The input data for the simulation job.

        Examples
        --------
        >>> job_manager.submit(Input(0.0, 1.0))

        Creates a job with the given parameters, logs it, and schedules it for submission
        through the job handling strategies.
        """

        job = Job(self._id_generator.generate_id(), x)
        self._handle_job(job, JobStatus.NOT_SUBMITTED)

    @staticmethod
    def _init_job_strategies() -> dict:
        """Initialises and returns a dictionary of job status to their corresponding handling strategies."""

        strategies = {
            JobStatus.COMPLETED: CompletedJobStrategy(),
            JobStatus.FAILED: FailedJobStrategy(),
            JobStatus.RUNNING: RunningJobStrategy(),
            JobStatus.SUBMITTED: SubmittedJobStrategy(),
            JobStatus.NOT_SUBMITTED: NotSubmittedJobStrategy(),
            JobStatus.CANCELLED: CancelledJobStrategy(),
        }

        return strategies

    def monitor(self, jobs: list[Job]):
        """
        Initiates or resumes monitoring of the specified jobs for status updates.

        Adds the provided list of jobs to the monitoring queue and starts or restarts
        the monitoring thread if it's not currently active. This ensures that all jobs
        are continuously monitored for status changes until they are completed or fail.

        Parameters
        ----------
        jobs : list[Job]
            A list of Job objects to be monitored.

        Notes
        ----
        This method is thread-safe and ensures that multiple calls to monitor jobs
        concurrently will not interfere with each other or duplicate monitoring efforts.

        Example
        -------
        >>> job_manager.monitor([job1, job2])

        Adds `job1` and `job2` to the monitoring queue and starts monitoring their statuses.
        """

        with self._lock:
            self._jobs.extend(jobs)
        if self._thread is None or not self._thread.is_alive():
            self._thread = Thread(target=self._monitor_jobs)
            self._thread.start()

    def _monitor_jobs(self):
        """Continuously monitor the status of jobs and handle their completion."""

        while self._jobs:
            sleep(self._polling_interval)
            with self._lock:
                jobs = self._jobs[:]
            for job in jobs:
                status = self._interface.get_job_status(job.id)
                self._handle_job(job, status)

    @property
    def interface(self):
        """Provides read-only access to the hardware interface used for job execution."""

        return self._interface

    @property
    def simulations_log(self):
        """Provides read-only access to the simulations log object for job recording and retrieval."""

        return self._simulations_log

    def remove_job(self, job: Job):
        """
        Removes a job from the internal list of jobs being monitored.

        This method ensures thread-safe removal of a job from the monitoring list,
        allowing for dynamic management of the jobs currently under monitoring by
        the JobManager.

        Parameters
        ----------
        job : Job
            The job instance to be removed from monitoring.

        Examples
        --------
        >>> job_manager.remove_job(job)

        This will remove the specified `job` from the JobManager's internal list, ceasing its monitoring.
        """
        with self._lock:
            self._jobs.remove(job)

    def _handle_job(self, job: Job, status: JobStatus):
        """Delegates handling of a job to the appropriate strategy based on its status."""

        strategy = self._job_strategies.get(status)
        if strategy:
            strategy.handle(job, self)


class JobStrategy(ABC):
    """
    Defines a template for job handling strategies in the simulation job management system.

    This abstract base class outlines the required interface for all job handling strategies.
    Concrete implementations of this class will define specific actions to be taken based
    on the job's current status.

    Methods
    -------
    handle(job: Job, job_manager: JobManager)
        Executes the strategy's actions for a given job within the context of the provided job manager.
    """

    @staticmethod
    @abstractmethod
    def handle(job: Job, job_manager: JobManager):
        """
        Handle a job according to the strategy's specific actions.

        This method should be implemented by subclasses to define how a job should be
        processed, based on its status or other criteria. It may involve submitting the job,
        updating its status, or performing cleanup actions.

        Parameters
        ----------
        job : Job
            The job to be handled, which contains the necessary information for processing.
        job_manager : JobManager
            The job manager instance, providing context and access to job management functionalities.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class CompletedJobStrategy(JobStrategy):
    """
    Implements the strategy for handling jobs that have completed execution.

    Upon invocation, this strategy retrieves the job's output from the simulation
    environment, updates the job's record in the simulations log to reflect its
    completion, and then removes the job from the JobManager's monitoring list.

    Parameters
    ----------
    job : Job
        The job that has completed its execution.
    job_manager : JobManager
        The manager responsible for overseeing the job's lifecycle.
    """

    @staticmethod
    def handle(job: Job, job_manager: JobManager):
        result = job_manager.interface.get_job_output(job.id)
        job_manager.simulations_log.insert_result(str(job.id), result)
        job_manager.simulations_log.update_job_status(str(job.id), JobStatus.COMPLETED)
        job_manager.remove_job(job)


class FailedJobStrategy(JobStrategy):
    """
    Strategy for handling jobs that have failed during execution.

    This strategy updates the job's status in the simulations log to FAILED and
    removes the job from the JobManager's list of active jobs. It encapsulates the
    actions to be taken when a job does not complete successfully.

    Parameters
    ----------
    job : Job
        The job that has failed.
    job_manager : JobManager
        The manager overseeing the job's lifecycle and responsible for its monitoring and logging.
    """

    @staticmethod
    def handle(job: Job, job_manager: JobManager):
        job_manager.simulations_log.update_job_status(str(job.id), JobStatus.FAILED)
        job_manager.remove_job(job)


class RunningJobStrategy(JobStrategy):
    """
    Strategy for handling jobs that are currently running.

    This strategy checks if a job's status is not already marked as RUNNING in the
    simulations log. If not, it updates the job's status to RUNNING. This ensures
    the job's current state is accurately reflected in the simulations log without
    unnecessarily updating the status of jobs already marked as running.

    Parameters
    ----------
    job : Job
        The job that is currently executing.
    job_manager : JobManager
        The manager responsible for the job's lifecycle, including monitoring and logging.
    """

    @staticmethod
    def handle(job: Job, job_manager: JobManager):
        if job_manager.simulations_log.get_job_status(str(job.id)) != JobStatus.RUNNING:
            job_manager.simulations_log.update_job_status(str(job.id), JobStatus.RUNNING)


class SubmittedJobStrategy(JobStrategy):
    """
    Strategy for handling jobs that have been submitted.

    Upon handling, this strategy updates the job's status in the simulations log to SUBMITTED
    and initiates monitoring of the job. This ensures that once a job is submitted, its
    status is accurately recorded, and the job is actively monitored for completion or failure.

    Parameters
    ----------
    job : Job
        The job that has been submitted for execution.
    job_manager : JobManager
        The manager overseeing the job's lifecycle, responsible for its submission, monitoring, and logging.
    """

    @staticmethod
    def handle(job: Job, job_manager: JobManager):
        job_manager.monitor([job])
        job_manager.simulations_log.update_job_status(str(job.id), JobStatus.SUBMITTED)


class NotSubmittedJobStrategy(JobStrategy):
    """
    Strategy for handling jobs that have not yet been submitted.

    This strategy attempts to submit the job with up to 5 retries, using
    exponential backoff and jitter to manage temporary issues like network congestion
    or service unavailability. If submission fails after all retries, the job's status.

    Parameters
    ----------
    job : Job
        The job to be submitted.
    job_manager : JobManager
        The manager responsible for job submission, monitoring, and logging.

    Notes
    -----
    This strategy uses exponential backoff to increase the delay between each retry attempt,
    and jitter to avoid thundering herd problems.
    """

    @staticmethod
    def handle(job: Job, job_manager: JobManager):
        retry_attempts = 0
        max_retries = 5
        initial_delay = 1
        max_delay = 32

        job_manager.simulations_log.add_new_record(job.data, str(job.id))

        while retry_attempts < max_retries:
            try:
                job_manager.interface.submit_job(job)
                job_manager.simulations_log.update_job_status(
                    str(job.id), JobStatus.SUBMITTED
                )
                job_manager.monitor([job])
                break
            except Exception as e:
                retry_attempts += 1

                if retry_attempts == max_retries:
                    print(
                        f"Max retry attempts reached for job {job.id}. Marking as FAILED_SUBMIT."
                    )
                    job_manager.simulations_log.update_job_status(
                        str(job.id), JobStatus.FAILED_SUBMIT
                    )
                    break

                delay = min(
                    initial_delay * (2**retry_attempts), max_delay
                )  # Exponential backoff
                jitter = random.uniform(0, 0.1 * delay)
                print(
                    f"Failed to submit job {job.id}: {e}. Retrying in {delay + jitter:.2f} seconds..."
                )
                sleep(delay + jitter)


class CancelledJobStrategy(JobStrategy):
    """
    Strategy for handling jobs that have been cancelled.

    Once a job is identified as cancelled, this strategy updates the job's status
    in the simulations log to CANCELLED and removes it from the list of jobs currently
    being monitored by the JobManager. This action signifies that no further processing
    is required or expected for the job.

    Parameters
    ----------
    job : Job
        The job that has been cancelled.
    job_manager : JobManager
        The manager overseeing the job's lifecycle, including its submission, monitoring,
        and status updates.
    """

    @staticmethod
    def handle(job: Job, job_manager: JobManager):
        job_manager.simulations_log.update_job_status(str(job.id), JobStatus.CANCELLED)
        job_manager.remove_job(job)


class JobIDGenerator:
    """
    A generator for unique job IDs, encapsulated within a JobId object, based on the
    current datetime down to the millisecond. This class provides a thread-safe
    mechanism to generate unique job IDs by ensuring that each ID corresponds to a
    unique point in time, formatted as 'YYYYMMDDHHMMSSfff', where 'fff' represents
    milliseconds.

    In scenarios where multiple IDs are requested within the same millisecond, this
    generator will wait until the next millisecond to generate a new ID, ensuring
    the uniqueness of each ID without relying on additional counters.

    Methods
    -------
    generate_id() -> JobId:
        Generates a unique JobId object representing the job ID, formatted as
        'YYYYMMDDHHMMSSfff', ensuring that each generated ID is unique to the
        millisecond.

    Examples
    --------
    >>> id_generator = JobIDGenerator()
    >>> job_id = id_generator.generate_id()
    >>> print(job_id)
    JobId('20240101123001005')
    """

    def __init__(self):
        """
        Initializes the JobIDGenerator, preparing it for generating unique job IDs.
        """
        self._lock = Lock()
        self._last_timestamp = None

    def generate_id(self) -> JobId:
        """
        Generates a unique job ID based on the current datetime down to the millisecond.
        If a request for a new ID occurs within the same millisecond as the previous ID,
        the method waits until the next millisecond to ensure uniqueness.

        Returns
        -------
        JobId
            A JobId object encapsulating a unique job ID, formatted as 'YYYYMMDDHHMMSSfff',
            ensuring uniqueness to the millisecond.

        Examples
        --------
        >>> id_generator = JobIDGenerator()
        >>> job_id = id_generator.generate_id()
        >>> print(job_id)
        JobId('20240101123001005')
        """
        with self._lock:
            while True:
                now = datetime.now()
                timestamp_str = now.strftime("%Y%m%d%H%M%S%f")[
                    :-3
                ]  # Convert to 'YYYYMMDDHHMMSSfff'

                if self._last_timestamp == timestamp_str:
                    sleep(0.001)  # Sleep for 1 millisecond to ensure uniqueness
                    continue
                else:
                    self._last_timestamp = timestamp_str
                    break

            return JobId(timestamp_str)
