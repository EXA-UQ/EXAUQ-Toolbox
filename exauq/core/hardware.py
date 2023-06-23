from abc import ABC, abstractmethod


class HardwareInterface(ABC):
    def __init__(self, hostname, username, password):
        self.hostname = hostname
        self.username = username
        self.password = password

    @abstractmethod
    def submit_job(self, job):
        pass

    @abstractmethod
    def get_job_status(self, job_id):
        pass

    @abstractmethod
    def get_job_output(self, job_id):
        pass
