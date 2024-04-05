from collections.abc import Sequence
from numbers import Real

from exauq.core.modelling import Input
from exauq.sim_management.hardware import HardwareInterface
from exauq.sim_management.jobs import Job
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

    # TODO: Return type just for illustration, not necessarily final API
    def submit(self, inputs: Sequence[Sequence[Real]]) -> dict[str, tuple[float, ...]]:
        return {str(n): tuple(map(float, x)) for n, x in enumerate(inputs)}

    # TODO: Return type just for illustration, not necessarily final API
    def status(self) -> dict[str, int]:
        return {"9999": 1}

    # TODO: Return type just for illustration, not necessarily final API
    def result(self) -> dict[str, int]:
        return {"9999": 1}
