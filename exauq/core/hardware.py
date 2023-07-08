from abc import ABC, abstractmethod
from typing import Optional

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
    def __init__(self, user: str, host: str, password: Optional[str] = None, key_filename: Optional[str] = None, ssh_config_path: Optional[str] = None):
        """
        SSH Interface to manage and submit jobs. Inherits from the HardwareInterface.

        Parameters
        ----------
        user : str
            The username to authenticate with the SSH server.
        host : str
            The hostname or IP address of the SSH server.
        password : str, optional
            The password to authenticate with the SSH server.
        key_filename : str, optional
            The path to the SSH private key file to authenticate with the SSH server.
        ssh_config_path : str, optional
            The path to the SSH configuration file.

        Raises
        ------
        ValueError
            If more than one method of authentication is provided.
        """

        # Check if more than one method is provided
        if sum([password is not None, key_filename is not None, ssh_config_path is not None]) > 1:
            raise ValueError(
                "Only one method of authentication should be provided. Please specify either password, key_filename, "
                "or ssh_config_path.")

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
