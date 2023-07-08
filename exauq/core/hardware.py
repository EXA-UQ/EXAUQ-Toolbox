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
    def __init__(self, user, host, password=None, key_filename=None, ssh_config_path=None):
        if password is not None:
            self.conn = Connection(f'{user}@{host}', connect_kwargs={"password": password})
        elif key_filename is not None:
            self.conn = Connection(f'{user}@{host}', connect_kwargs={"key_filename": key_filename})
        elif ssh_config_path is not None:
            from fabric import Config
            ssh_config = Config(overrides={'ssh_config_path': ssh_config_path})
            self.conn = Connection(host, config=ssh_config)
        else:
            self.conn = Connection(f'{user}@{host}')  # Defaults to SSH agent if no password or key is provided

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
