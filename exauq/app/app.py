from collections.abc import Sequence
from numbers import Real
from typing import Any, Optional, Union

from exauq.core.modelling import Input
from exauq.sim_management.hardware import HardwareInterface, JobStatus
from exauq.sim_management.jobs import Job, JobId
from exauq.sim_management.simulators import JobManager, SimulationsLog
from exauq.sim_management.types import FilePath


class App:
    """
    Provides a high-level interface for submitting simulation jobs and managing their statuses.

    This class acts as a facade to the more complex simulation management components, offering
    a simplified interface for submitting batches of simulation jobs and querying job statuses.
    It initialises and coordinates interactions between the hardware interface, the simulations log,
    and the job manager.

    Parameters
    ----------
    interface : HardwareInterface
        The hardware interface through which simulation jobs will be executed.
    input_dim : int
        The dimensionality of the input data for simulations.
    simulations_log_file : FilePath, optional
        Path to the file where simulation job logs will be stored. Defaults to "simulations.csv".

    Methods
    -------
    submit(inputs: Sequence[Sequence[Real]]) -> tuple[Job]
        Submits a batch of simulation jobs based on the provided inputs.
    get_jobs(jobs: Sequence[Union[str, JobId, int]] = None, statuses: Sequence[JobStatus] = None) -> list[dict[str, Any]]
        Retrieves information about jobs, optionally filtered by job IDs or statuses.
    shutdown()
        Cleanly terminates the job monitoring process and ensures all resources are properly released.
    """

    def __init__(
        self,
        interface: HardwareInterface,
        input_dim: int,
        simulations_log_file: FilePath = "simulations.csv",
    ):
        """
        Initialises the application with the specified hardware interface, input dimensionality for simulations,
        and path to the simulations log file.
        """
        self._sim_log_path = simulations_log_file
        self._input_dim = input_dim
        self._interface = interface
        self._sim_log = SimulationsLog(self._sim_log_path, self._input_dim)
        self._job_manager = JobManager(
            simulations_log=self._sim_log,
            interface=self._interface,
            polling_interval=10,
            wait_for_pending=False,
        )

    def submit(self, inputs: Sequence[Sequence[Real]]) -> tuple[Job]:
        """
        Submits a batch of simulation jobs to the job manager based on the provided input sequences.

        Each input sequence is converted into an Input object and then submitted as a new simulation
        job through the JobManager. This method is ideal for bulk submission of jobs to utilize
        hardware resources efficiently.

        Parameters
        ----------
        inputs : Sequence[Sequence[Real]]
            A sequence of input sequences, where each inner sequence represents the input parameters
            for a single simulation job.

        Returns
        -------
        tuple[Job]
            A tuple of the Job objects created for each submitted job, allowing for further interaction
            or status checking.

        Examples
        --------
        >>> jobs = app.submit([(1.0, 2.0), (3.0, 4.0)])
        >>> for job in jobs:
        ...     print(job.id)

        This demonstrates submitting two jobs to the simulation environment with different input parameters.
        """

        submitted_jobs = []
        for inp in inputs:
            submitted_jobs.append(self._job_manager.submit(Input(*inp)))

        return tuple(submitted_jobs)

    def get_jobs(
        self,
        job_ids: Sequence[Union[str, JobId, int]] = None,
        n_most_recent: int = None,
        statuses: Sequence[JobStatus] = None,
        result_filter: Optional[bool] = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieves records of simulation jobs, optionally filtered by specific job IDs or job statuses.

        This method queries the simulations log to fetch details of jobs. It can be used to get the
        current status of jobs, their input parameters, or outcomes. Filtering by job IDs or statuses
        allows for more targeted information retrieval.

        Parameters
        ----------
        job_ids : Sequence[Union[str, JobId, int]], optional
            A sequence of job identifiers (IDs, JobId objects, or numerical IDs) to specifically
            retrieve records for. If None, information on all jobs will be returned.
        statuses : Sequence[JobStatus], optional
            A sequence of job statuses to filter the retrieved job records by. If None, jobs
            of all statuses will be included in the result.

        Returns
        -------
        list[dict[str, Any]]
            A list of dictionaries, where each dictionary contains details of a single job,
            such as its ID, status, input parameters, and outcomes.

        Examples
        --------
        >>> jobs_info = app.get_jobs(statuses=[JobStatus.COMPLETED])
        >>> for info in jobs_info:
        ...     print(info['job_id'], info['status'])

        This example retrieves and prints the IDs and statuses of all jobs that have been completed.
        """

        if n_most_recent is not None and n_most_recent == 0:
            return []
        elif n_most_recent is not None and n_most_recent < 0:
            raise ValueError("'n_most_recent' must be non-negative")
        else:
            jobs = sorted(
                self._filter_records(
                    self._sim_log.get_records(job_ids, statuses),
                    result_filter=result_filter,
                ),
                key=lambda x: str(x["job_id"]),
            )
            if n_most_recent is not None:
                return jobs[-n_most_recent:]
            else:
                return jobs

    def _filter_records(
        self, records: list[dict[str, Any]], result_filter: Optional[bool] = None
    ):
        if result_filter is None:
            return records
        elif result_filter:
            return filter(lambda x: x["output"] is not None, records)
        else:
            return filter(lambda x: x["output"] is None, records)

    def shutdown(self):
        """
        Initiates a clean shutdown process for the application, particularly the job monitoring
        mechanism.

        This method is critical for ensuring that the application exits cleanly without leaving
        any orphaned threads or unresolved resources. It delegates the shutdown process to the
        job manager, which terminates its monitoring thread and ensures all resources are properly
        released.

        Examples
        --------
        >>> app.shutdown()

        Demonstrates initiating a shutdown of the application's job management and monitoring systems.
        """
        self._job_manager.shutdown()
