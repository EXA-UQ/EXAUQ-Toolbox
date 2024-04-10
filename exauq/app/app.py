from collections.abc import Sequence
from numbers import Real
from typing import Any, Union

from exauq.core.modelling import Input
from exauq.sim_management.hardware import HardwareInterface, JobStatus
from exauq.sim_management.jobs import Job, JobId
from exauq.sim_management.simulators import JobManager, SimulationsLog
from exauq.sim_management.types import FilePath


class App:
    def __init__(
        self,
        interface: HardwareInterface,
        input_dim: int,
        simulations_log_file: FilePath = "simulations.csv",
    ):
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
        submitted_jobs = []
        for inp in inputs:
            submitted_jobs.append(self._job_manager.submit(Input(*inp)))

        return tuple(submitted_jobs)

    def get_jobs(
        self,
        jobs: Sequence[Union[str, JobId, int]] = None,
        statuses: Sequence[JobStatus] = None,
    ) -> list[dict[str, Any]]:

        return self._sim_log.get_records(jobs, statuses)
