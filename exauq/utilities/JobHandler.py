import typing
from abc import ABC, abstractmethod

class JobHandler(ABC):
    """
     Class describing a job handler
    """
    def __init__(self):
        self.handler = None

    @abstractmethod
    def job_submit(self, command, host, username) -> typing.Tuple[str]:
        """
        Method that runs a command and returns the process/job id, and the
        stdout and stderr file names.
        """
        pass

    @abstractmethod
    def poll(self, job_id, host, username) -> str:
        """
        Method that polls a command
        """
        pass