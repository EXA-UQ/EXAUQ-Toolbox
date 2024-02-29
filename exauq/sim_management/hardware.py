import getpass
import io
import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

from fabric import Config, Connection
from paramiko.ssh_exception import AuthenticationException, SSHException

from exauq.sim_management.jobs import Job, JobId


class JobStatus(Enum):
    """Represents the statuses of jobs that can arise when running jobs.

    It is possible for multiple statuses hold at the same time. For example, a job
    can both have been submitted (`SUBMITTED`) and also have been run and exited with
    error (`FAILED`).
    """

    COMPLETED = "Completed"
    """A job has run to completion on a remote machine without error. Has the value
    'Completed'."""

    FAILED = "Failed"
    """A job has been run on a remote machine but has exited with an error. Has the
    value 'Failed'."""

    RUNNING = "Running"
    """A job is running on a remote machine but has not yet finished. Has the value
    'Running'."""

    SUBMITTED = "Submitted"
    """A job has been successfully sent to a remote machine. Has the value 'Submitted'."""

    NOT_SUBMITTED = "Not submitted"
    """A job has been set up locally but has not yet been submitted to a remote
    machine. Has the value 'Not submitted'."""

    CANCELLED = "Cancelled"
    """A job has been cancelled from a locally issued request or intervention. Has the
    value 'Cancelled'."""


class HardwareInterface(ABC):
    """
    Abstract base class for a hardware interface.

    This class defines the abstract methods that any hardware interface should implement,
    providing a consistent API for interacting with different types of hardware resources.

    The HardwareInterface class is not specific to any type of hardware or infrastructure.
    It can be extended to provide interfaces for various types of hardware resources,
    such as supercomputers, GPU clusters, servers, personal laptops, or even potatoes!
    Whether the hardware is local or remote is also abstracted away by this interface.

    The goal is to provide a unified way to submit jobs, query job status, fetch job output,
    cancel jobs, and wait for jobs across all types of hardware resources. This enables
    writing hardware-agnostic code for running simulations or performing other computational tasks.

    Implementations should provide the following methods:
    - submit_job
    - get_job_status
    - get_job_output
    - cancel_job
    """

    @abstractmethod
    def submit_job(self, job: Job):
        raise NotImplementedError

    @abstractmethod
    def get_job_status(self, job_id: JobId):
        raise NotImplementedError

    @abstractmethod
    def get_job_output(self, job_id: JobId):
        raise NotImplementedError

    @abstractmethod
    def cancel_job(self, job_id: JobId):
        raise NotImplementedError


class SSHInterface(HardwareInterface, ABC):
    """
    SSH Interface to manage and submit jobs. Inherits from the HardwareInterface.

    The SSHInterface class provides an interface for interacting with hardware over SSH. It can
    authenticate using either a key file, an SSH config path, or via an SSH agent. If none of
    these methods are provided, it will prompt for a password.

    Parameters
    ----------
    user : str
        The username to authenticate with the SSH server.
    host : str
        The hostname or IP address of the SSH server.
    key_filename : str, optional
        The path to the SSH private key file to authenticate with the SSH server.
    ssh_config_path : str, optional
        The path to the SSH configuration file.
    use_ssh_agent : bool, optional
        If ``True``, use SSH agent for authentication. Defaults to ``False``.
    max_attempts : int, optional
        The number of authentication attempts allowed. Defaults to ``3``.

    Raises
    ------
    ValueError
        If more than one method of authentication is provided.
    """

    def __init__(
        self,
        user: str,
        host: str,
        key_filename: Optional[str] = None,
        ssh_config_path: Optional[str] = None,
        use_ssh_agent: Optional[bool] = False,
        max_attempts: int = 3,
    ):
        self.max_attempts = max_attempts

        # Check if more than one method is provided
        if (
            sum(
                [
                    key_filename is not None,
                    ssh_config_path is not None,
                    use_ssh_agent,
                ]
            )
            > 1
        ):
            raise ValueError(
                "Only one method of authentication should be provided. Please specify either "
                "'key_filename', 'ssh_config_path' or set 'use_ssh_agent' to True."
            )

        if key_filename is not None:
            self._conn = Connection(
                f"{user}@{host}", connect_kwargs={"key_filename": key_filename}
            )
        elif ssh_config_path is not None:
            ssh_config = Config(overrides={"ssh_config_path": ssh_config_path})
            self._conn = Connection(host, config=ssh_config)
        elif use_ssh_agent:
            self._conn = Connection(f"{user}@{host}")
        else:
            self._init_with_password(user, host)

        self._check_connection()

    def _check_connection(self):
        try:
            self._conn.run('echo "Testing connection"', hide=True)
            print(f"Connection to {self._conn.original_host} established.")
            return

        except Exception as e:
            message = f"Could not connect to {self._conn.original_host}: {str(e)}"
            raise Exception(message) from None

    def _init_with_password(self, user: str, host: str):
        for attempt in range(1, self.max_attempts + 1):
            password = getpass.getpass(prompt=f"Password for {user}@{host}: ")
            try:
                self._conn = Connection(
                    f"{user}@{host}", connect_kwargs={"password": password}
                )
                # Check connection by running a simple command
                self._conn.run('echo "Testing connection"', hide=True)
                return

            except AuthenticationException:  # Catch the specific exception
                if attempt < self.max_attempts:  # Don't say this on the last attempt
                    print("Failed to authenticate. Please try again.")
                else:
                    print("Maximum number of attempts exceeded.")
                    raise
            except SSHException as e:
                print(f"Could not connect to {self._conn.original_host}: {str(e)}")
                raise  # Re-raise the exception

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._conn.close()


class RemoteServerScript(SSHInterface):
    def __init__(
        self,
        user: str,
        host: str,
        program: str,
        script_path: str,
        config_path: str,
        script_output_path: str,
        stdout_path: str,
        key_filename: Optional[str] = None,
        ssh_config_path: Optional[str] = None,
        use_ssh_agent: Optional[bool] = False,
        max_attempts: int = 3,
    ):
        super().__init__(
            user, host, key_filename, ssh_config_path, use_ssh_agent, max_attempts
        )
        self._program = program
        self._script_path = script_path
        self._config_path = config_path
        self._script_output_path = script_output_path
        self._stdout_path = stdout_path
        self._job_log = dict()

    def submit_job(self, job: Job) -> None:
        wrapper_script = "\n".join(
            [
                f"{self._program} {self._script_path} {self._config_path} >> {self._stdout_path} 2>&1 &",
                "job_pid=$!",
                self._make_process_identifier_command("${job_pid}"),
            ]
        )

        # TODO: use mktemp remotely?
        wrapper_script_path = os.path.dirname(self._script_path) + "/job_wrapper.sh"
        self._make_text_file_on_remote(wrapper_script, wrapper_script_path)
        remote_id = self._run_remote_command(f"bash {wrapper_script_path}")
        remote_id_components = self._parse_process_identifier(remote_id)
        self._job_log[job.id] = {
            "status": JobStatus.SUBMITTED,
            "remote_id": remote_id,
            "pid": remote_id_components["pid"],
            "start_time": remote_id_components["start_time"],
            "script_output_path": self._script_output_path,
            "output": None,
        }
        return None

    def _make_process_identifier_command(self, pid: str) -> str:
        """Make a shell command for getting a string that uniquely identifies a remote
        process."""

        return f'echo "$(ps -p {pid} -o user=),$(ps -p {pid} -o pid=),$(ps -p {pid} -o lstart=)"'

    def _parse_process_identifier(self, process_identifier: str) -> dict[str, str]:
        """Extract the user, process ID and start time of a process.

        The `process_identifier` should be as output by the command made by the
        `_make_process_identifier_command` method.
        """

        user, pid, start_time = process_identifier.split(",")
        return {"user": user, "pid": pid, "start_time": start_time}

    def _make_text_file_on_remote(self, file_contents: str, target_path: str) -> str:
        """Make a text file on the remote machine with a given string as contents."""

        _ = self._conn.put(
            io.StringIO(file_contents),
            remote=target_path,
        )
        return None

    def _run_remote_command(self, command: str) -> str:
        """Run a shell command and return the standard output."""

        res = self._conn.run(command, hide=True)
        return str(res.stdout).strip()

    def get_job_status(self, job_id: JobId) -> JobStatus:
        if not self._job_has_been_submitted(job_id):
            return JobStatus.NOT_SUBMITTED
        else:
            self._update_status_from_remote(job_id)
            return self._job_log[job_id]["status"]

    def _job_has_been_submitted(self, job_id: JobId) -> bool:
        """Whether a job with the given ID has been submitted."""

        return job_id in self._job_log

    def _update_status_from_remote(self, job_id: str) -> None:
        """Update the status of a job based on the remote status of the corresponding
        process."""

        if self._remote_job_is_running(job_id):
            self._job_log[job_id]["status"] = JobStatus.RUNNING
        else:
            output = self.get_job_output(job_id)
            if output is not None:
                self._job_log[job_id]["status"] = JobStatus.COMPLETED

        return None

    def _retrieve_output(self, remote_path: str) -> Optional[float]:
        """Get the output of a simulation from the remote server."""

        with io.BytesIO() as buffer:
            try:
                _ = self._conn.get(remote_path, local=buffer)
            except FileNotFoundError:
                return None

            contents = buffer.getvalue().decode(encoding="utf-8")

        # TODO: raise custom exception if cannot cast to float?
        return float(contents.strip())

    def _remote_job_is_running(self, job_id) -> bool:
        """Whether the remote process of a given job is running."""

        pid = self._job_log[job_id]["pid"]
        process_identifier_command = self._make_process_identifier_command(pid)
        process_identifier = self._run_remote_command(process_identifier_command)
        return self._job_log[job_id]["remote_id"] == process_identifier

    def get_job_output(self, job_id: JobId) -> Optional[float]:
        if not self._job_has_been_submitted(job_id):
            return None

        elif self._job_log[job_id]["output"] is not None:
            return self._job_log[job_id]["output"]

        else:
            output_path = self._job_log[job_id]["script_output_path"]
            self._job_log[job_id]["output"] = self._retrieve_output(output_path)
            return self._job_log[job_id]["output"]

    def cancel_job(self, job_id: JobId):
        pass
