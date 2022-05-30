from abc import ABC, abstractmethod

class JobHandler(ABC):
    """
     Class describing a job handler
    """
    def __init__(self):
        self.handler = None

    @abstractmethod
    def job_submit(self) -> str:
        """
        Method that runs a command and return the process id
        """
        pass

    @abstractmethod
    def poll(self, proc_id) -> str:
        """
        Method that polls a command
        """
        pass