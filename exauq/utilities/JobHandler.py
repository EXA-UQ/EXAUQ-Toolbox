from abc import ABC, abstractmethod


class JobHandler(ABC):
    """
    Class describing a job handler
    """

    ROOT_RUN_DIR = "exauq-run"

    def __init__(self, host: str, user: str, run_local=False) -> None:
        self.host = host
        self.user = user
        self.run_local = run_local
        self.sim_dir = None
        self.run_process = None
        self.poll_process = None
        self.job_id = None
        self.job_status = None
        self.submit_time = None
        self.last_poll_time = None

    @abstractmethod
    def submit_job(self, sim_id: str, command: str) -> None:
        """
        Method that submits a job remotely.
        """
        pass

    @abstractmethod
    def poll_job(self, sim_id: str, job_id: str) -> None:
        """
        Method that polls the job/process with given id
        """
        pass
