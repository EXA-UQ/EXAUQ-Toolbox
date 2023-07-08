from abc import ABC, abstractmethod

from fabric import Connection


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


class SSHInterface(HardwareInterface):
    def __init__(self, user, host, password):
        self.conn = Connection(f'{user}@{host}', connect_kwargs={"password": password})

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
