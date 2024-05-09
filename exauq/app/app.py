from collections.abc import Sequence
from numbers import Real
from typing import Any, Optional, Union

from exauq.core.modelling import Input
from exauq.sim_management.hardware import HardwareInterface, JobStatus
from exauq.sim_management.jobs import Job, JobId
from exauq.sim_management.simulators import (
    InvalidJobStatusError,
    JobManager,
    SimulationsLog,
    UnknownJobIdError,
)
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

    @property
    def input_dim(self) -> int:
        """(Read-only) The dimensionality of the input data for simulations."""

        return self._input_dim

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

    def cancel(self, job_ids: Sequence[Union[str, JobId, int]]) -> list[dict[str, Any]]:

        report = {
            "cancelled_jobs": [],
            "non_existent_jobs": [],
            "terminated_jobs": [],
        }
        for job_id in job_ids:
            try:
                cancelled_job = self._job_manager.cancel(job_id)
                report["cancelled_jobs"].append(
                    {
                        "job_id": cancelled_job.id,
                        "input": cancelled_job.data,
                        "status": JobStatus.CANCELLED,
                    }
                )
            except UnknownJobIdError:
                report["non_existent_jobs"].append(job_id)
            except InvalidJobStatusError:
                report["terminated_jobs"].append(job_id)

        return report

    def get_jobs(
        self,
        job_ids: Sequence[Union[str, JobId, int]] = None,
        n_most_recent: Optional[int] = None,
        statuses: Optional[Sequence[JobStatus]] = None,
        result_filter: Optional[bool] = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieves records of simulation jobs, with optional filtering.

        This method queries the simulations log to fetch details of jobs, subject to
        optional filters on job ID, job status, the number of jobs (from the most recent)
        and whether there is a simulation output.

        Parameters
        ----------
        job_ids : Sequence[Union[str, JobId, int]], optional
            (Default: None) IDs of the jobs to retrieve records for. If ``None``, retrieve
            all records from the log, subject to other filters.
        n_most_recent : int, optional
            (Default: None) The number of job records to return, counting back from the
            most recent in terms of job ID. If ``None`` then do not restrict the number of
            records returned from the log.
        statuses : Sequence[JobStatus], optional
            (Default: None) Job statuses to filter on, so that the records returned will
            only be for jobs with one of these statuses. If ``None``, then do not restrict
            the records returned from the log by status.
        result_filter : bool, optional
            (Default: None) Whether to get only jobs that have a simulation output
            (``True``), get only jobs that don't have a simulation output (``False``), or
            whether not to restrict the records returned from the log by presence of
            simulation output (``None``).

        Returns
        -------
        list[dict[str, Any]]
            Records of jobs from the simulations log that meet the specified filtering
            criteria. The keys and values of the dictionary are

            * 'job_id': the ``JobID`` for the job.
            * 'status': the ``JobStatus`` for the job.
            * 'input': the simulation input, as an ``Input``.
            * 'output': the simulation output as a ``float`` if available, or ``None``
              if not.

        Examples
        --------
        >>> jobs_info = app.get_jobs(statuses=[JobStatus.COMPLETED])
        >>> for info in jobs_info:
        ...     print(info['job_id'], info['status'])

        This example retrieves and prints the IDs and statuses of all jobs that have been
        completed.
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
