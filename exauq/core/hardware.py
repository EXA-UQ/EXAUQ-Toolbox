from abc import ABC, abstractmethod


class HardwareInterface(ABC):

    @abstractmethod
    def submit_job(self, job):
        pass

    @abstractmethod
    def get_job_status(self, job_id):
        pass

    @abstractmethod
    def get_job_output(self, job_id):
        pass

    @abstractmethod
    def cancel_job(self, job_id):
        pass

    @abstractmethod
    def wait_for_job(self, job_id):
        pass
