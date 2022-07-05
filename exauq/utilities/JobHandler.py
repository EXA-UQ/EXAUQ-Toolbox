from abc import ABC, abstractmethod

class JobHandler(ABC):
    """
     Class describing a job handler
    """
    def __init__(self, host: str, user: str) -> None:
        self.host = host
        self.user = user
        self.run_process = None
        self.poll_process = None
        self.job_id = None
        self.job_status = None

    @abstractmethod
    def submit_job(self, sim_id: str, command: str) -> None:
        """
        Method that submits a job remotely.
        """
        pass

    @abstractmethod
    def get_jobid(self) -> None:
        """
        Method that returns the job id of submitted job.
        """
        pass

    @abstractmethod
    def poll_job(self, sim_id: str, job_id: str) -> None:
        """
        Method that polls the job/process with given id
        """
        pass