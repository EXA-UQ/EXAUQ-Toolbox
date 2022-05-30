from abc import ABC, abstractmethod

class JobHandler(ABC):
    """
     Class describing a job handler
    """
    def __init__(self, host: str, user: str) -> None:
        self.host = host
        self.user = user

    @abstractmethod
    def submit_job(self, sim_id: str, command: str) -> str:
        """
        Method that runs a command and returns the process/job id.
        """
        pass

    @abstractmethod
    def poll_job(self, sim_id: str, job_id: str) -> str:
        """
        Method that polls a job/process with given id
        """
        pass