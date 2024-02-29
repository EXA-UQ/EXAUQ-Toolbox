import getpass
import io
import json
import os
from abc import ABC, abstractmethod
from collections.abc import Sequence
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
    """Interface for running a simulation script on a remote server over SSH.

    This interface is designed to invoke simulation code on a remote server with the
    following command:

    ```
    <program> <script_path> path/to/config.json
    ```

    Here, ``<program>`` is a program such as `bash`, `python`, `Rscript`, etc. The script
    is expected to take a single JSON config file as input. This config file consists of a
    single object with keys "input_file" and "output_file", which give paths to an
    input csv file for the simulator and an output file for the simulator to write its
    output to.

    Objects of this class will take care of creating the necessary JSON config file and
    input file from a `Job`, uploading these to a job-specific subdirectory of a remote
    'workspace' directory. This job-specific directory is also where the output of the
    simulator script is written to, along with a file capturing standard output and
    standard error from running the script. Note that any required intermiary directories
    are created on the server.

    If `key_filename` and `ssh_config_path` are not provided and `use_ssh_agent` is
    ``False`` then a prompt for a password from standard input will be issued each time a
    connection is made to the server.

    Parameters
    ----------
    user : str
        The username to authenticate with the SSH server.
    host : str
        The hostname or IP address of the SSH server.
    program : str
        The program to run on the server.
    script_path : str
        The path to the script on the server to run with `program`.
    remote_workspace_dir : str, optional
        (Default: None) A remote path a directory where job-specific subdirectories should
        be created. Relative paths will be relative to the default working directory
        for a new SSH session (usually the user's home directory). If ``None`` then the
        directory containing the script in `script_path` will be used.
    key_filename : str, optional
        (Default: None) The path to an SSH private key file to authenticate with the SSH
        server. The key file must be unencrypted.
    ssh_config_path : str, optional
        (Default: None) The path to an SSH configuration file.
    use_ssh_agent : bool, optional
        (Default: False) If ``True``, use a running SSH agent for authentication.
    max_attempts : int, optional
        (Default: 3) The number of authentication attempts allowed.

    Raises
    ------
    ValueError
        If more than one method of authentication is provided.
    """

    def __init__(
        self,
        user: str,
        host: str,
        program: str,
        script_path: str,
        remote_workspace_dir: Optional[str] = None,
        key_filename: Optional[str] = None,
        ssh_config_path: Optional[str] = None,
        use_ssh_agent: Optional[bool] = False,
        max_attempts: int = 3,
    ):
        super().__init__(
            user, host, key_filename, ssh_config_path, use_ssh_agent, max_attempts
        )
        self._user = user
        self._host = host
        self._program = program
        self._script_path = script_path
        self._remote_workspace_dir = (
            remote_workspace_dir
            if remote_workspace_dir is not None
            else os.path.dirname(self._script_path)
        )
        self._job_log = dict()

    def submit_job(self, job: Job) -> None:
        """Submit a job for the simulation code.

        Upon submission, a new subdirectory of the remote workspace directory supplied
        at this object's initialisation will be created for the job, using the job's ID
        as the directory name. A JSON config file for the script will be created and
        uploaded to this directory, along with a simulator input csv file containing the
        data from the job. The content of the JSON config file will have the following
        form:

        ```
        {"input_file": "path/to/job_dir/input.csv", "output_file": "path/to/job_dir/output.txt"}
        ```

        Finally, a simple shell script wrapping the main command for running the simulator
        will be uploaded: this is what gets run to start the job on the server.

        Parameters
        ----------
        job : Job
            A job containing the data to run the simulation code with.

        Raises
        ------
        HardwareInterfaceFailure
            If there were problems connecting to the server, making files / directories on
            the server or other such server-related problems.
        """
        job_remote_dir = self._remote_workspace_dir + f"/{job.id}"
        self._make_directory_on_remote(job_remote_dir)
        script_input_path = self._make_job_input_file(job.data, job_remote_dir)
        script_config = {
            "input_file": script_input_path,
            "output_file": job_remote_dir + "/output.txt",
        }
        config_path = self._make_job_config_file(script_config, job_remote_dir)
        stdout_path = job_remote_dir + f"/{os.path.basename(self._script_path)}.out"
        wrapper_script = "\n".join(
            [
                f"{self._program} {self._script_path} {config_path} >> {stdout_path} 2>&1 &",
                "job_pid=$!",
                self._make_process_identifier_command("${job_pid}"),
            ]
        )

        wrapper_script_path = job_remote_dir + "/job_wrapper.sh"
        self._make_text_file_on_remote(wrapper_script, wrapper_script_path)
        remote_id = self._run_remote_command(f"bash {wrapper_script_path}")
        remote_id_components = self._parse_process_identifier(remote_id)
        self._job_log[job.id] = {
            "status": JobStatus.SUBMITTED,
            "remote_id": remote_id,
            "pid": remote_id_components["pid"],
            "start_time": remote_id_components["start_time"],
            "job_remote_dir": job_remote_dir,
            "script_output_path": script_config["output_file"],
            "output": None,
        }
        return None

    def _make_job_config_file(self, config: dict[str, str], job_remote_dir: str) -> str:
        """Make a simulation script input JSON configuration file on the server and
        return the path to this file."""

        config_file_path = job_remote_dir + "/config.json"
        self._make_text_file_on_remote(json.dumps(config), config_file_path)
        return config_file_path

    def _make_job_input_file(self, data: Sequence, job_remote_dir: str) -> str:
        """Make a csv file on the server containing input data for the simulation code."""

        input_file_path = job_remote_dir + "/input.csv"
        data_str = ",".join(map(str, data)) + "\n"
        self._make_text_file_on_remote(data_str, input_file_path)
        return input_file_path

    def _make_process_identifier_command(self, pid: str) -> str:
        """Make a shell command for getting a string that uniquely identifies a remote
        process."""

        return f'echo "$(ps -p {pid} -o user=),$(ps -p {pid} -o pid=),$(ps -p {pid} -o lstart=)"'

    def _parse_process_identifier(self, process_identifier: str) -> dict[str, str]:
        """Extract the user, process ID and start time of a process from a process
        identifier.

        The `process_identifier` should be a comma-delimited string of the form
        '<user>,<pid>,<start_time>'.
        """

        user, pid, start_time = process_identifier.split(",")
        return {"user": user, "pid": pid, "start_time": start_time}

    def _make_directory_on_remote(self, path: str) -> None:
        """Make a directory at the given path on the remote machine.

        Will recursively create intermediary directories as required.
        """
        try:
            _ = self._run_remote_command(f"mkdir -p {path}")
        except Exception as e:
            raise HardwareInterfaceFailureError(
                f"Could not make directory {path} for {self._user}@{self._host}: {e}"
            )
        return None

    def _make_text_file_on_remote(self, file_contents: str, target_path: str) -> str:
        """Make a text file on the remote machine with a given string as contents."""

        try:
            _ = self._conn.put(
                io.StringIO(file_contents),
                remote=target_path,
            )
        except Exception as e:
            raise HardwareInterfaceFailureError(
                f"Could not create text file at {target_path} for "
                f"{self._user}@{self._host}: {e}"
            )
        return None

    def _run_remote_command(self, command: str) -> str:
        """Run a shell command and return the resulting contents of standard output.

        The contents of standard output is stripped of leading/trailing whitespace before
        returning."""

        res = self._conn.run(command, hide=True)
        return str(res.stdout).strip()

    def get_job_status(self, job_id: JobId) -> JobStatus:
        """Get the status of a job with given job ID.

        Any jobs that have not been submitted with `submit_job` will return a status of
        `JobStatus.NOT_SUBMITTED`.

        A job that has successfully been started on the server will have a status of
        `JobStatus.RUNNING` (which, in this case, is equivalent to `JobStatus.SUBMITTED`).
        The status will remain as `JobStatus.RUNNING` until the corresponding remote
        process has stopped, in which case the status of the job will be
        `JobStatus.COMPLETED` if an output file from the simulator has been created or
        `JobStatus.FAILED` if not. (In particular, not that the exit code of the
        simulator script is not taken into account when determining whether a job has
        finished successfully or not.)

        Parameters
        ----------
        job_id : JobId
            The ID of the job to check the status of.

        Returns
        -------
        JobStatus
            The status of the job.
        """
        if not self._job_has_been_submitted(job_id):
            return JobStatus.NOT_SUBMITTED
        else:
            self._update_status_from_remote(job_id)
            return self._job_log[job_id]["status"]

    def _job_has_been_submitted(self, job_id: JobId) -> bool:
        """Whether a job with the given ID has been submitted."""

        return job_id in self._job_log

    def _update_status_from_remote(self, job_id: str) -> None:
        """Update the status of a job based on the status of the corresponding process on
        the server."""

        if self._remote_job_is_running(job_id):
            self._job_log[job_id]["status"] = JobStatus.RUNNING
        else:
            output = self.get_job_output(job_id)
            if output is not None:
                self._job_log[job_id]["status"] = JobStatus.COMPLETED
            else:
                self._job_log[job_id]["status"] = JobStatus.FAILED

        return None

    def _remote_job_is_running(self, job_id) -> bool:
        """Whether the remote process of a given job is running."""

        pid = self._job_log[job_id]["pid"]
        process_identifier_command = self._make_process_identifier_command(pid)
        process_identifier = self._run_remote_command(process_identifier_command)
        return self._job_log[job_id]["remote_id"] == process_identifier

    def get_job_output(self, job_id: JobId) -> Optional[float]:
        """Get the simulator output for a job with the given ID.

        This is read from the contents of the simulator output file for the job, located
        in the job's remote directory. It is expected that the contents of this output
        file will be a single floating point number.

        Parameters
        ----------
        job_id : JobId
            The ID of the job to get the simulator output for.

        Returns
        -------
        Optional[float]
            The output of the simulator, if the job has completed successfully, or else
            ``None``.

        Raises
        ------
        HardwareInterfaceFailure
            If there were problems connecting to the server or retrieving the simulator
            output.
        SimulatorOutputParsingError
            If the output of the simulator cannot be parsed as a single floating point
            number.
        """
        if not self._job_has_been_submitted(job_id):
            return None

        elif self._job_log[job_id]["output"] is not None:
            return self._job_log[job_id]["output"]

        else:
            output_path = self._job_log[job_id]["script_output_path"]
            output = self._retrieve_output(output_path)
            try:
                self._job_log[job_id]["output"] = float(output)
            except ValueError:
                raise SimulatorOutputParsingError(
                    f"Could not parse simulator output {output} for job ID {job_id} as a "
                    "float."
                )
            return self._job_log[job_id]["output"]

    def _retrieve_output(self, remote_path: str) -> Optional[str]:
        """Get the output of a simulation from the remote server."""

        with io.BytesIO() as buffer:
            try:
                _ = self._conn.get(remote_path, local=buffer)
            except FileNotFoundError:
                return None
            except Exception as e:
                raise HardwareInterfaceFailureError(
                    f"Could not retrieve output of script {self._script_path} from file "
                    f"{remote_path}: {e}"
                )

            contents = buffer.getvalue().decode(encoding="utf-8")

        return contents.strip()

    def cancel_job(self, job_id: JobId):
        pass

    def delete_remote_job_dir(self, job_id: JobId) -> None:
        """Delete the remote directory corresponding to a given job ID.

        This will recursively delete all the contents of the directory, invoking
        ``rm -r`` on it.

        Parameters
        ----------
        job_id : JobId
            The ID of the job whose remote directory should be deleted.

        Raises
        ------
        ValueError
            If the supplied job ID has not been submitted since this object was initialised.
        HardwareInterfaceFailure
            If there were problems connecting to the server or deleting the directory.
        """

        if job_id not in self._job_log:
            raise ValueError(
                f"Job ID {job_id} not submitted since initialisation of this object."
            )
        else:
            job_remote_dir = self._job_log[job_id]["job_remote_dir"]
            deletion_cmd = f"rm -r {job_remote_dir}"
            try:
                _ = self._run_remote_command(deletion_cmd)
            except Exception as e:
                raise HardwareInterfaceFailureError(
                    f"Could not delete remote folder {job_remote_dir} for "
                    f"{self._user}@{self._host}: {e} "
                )

            return None


class HardwareInterfaceFailureError(Exception):
    """Raised when an error was encounted when running a command or communicating with a
    machine."""

    pass


class SimulatorOutputParsingError(Exception):
    """Raised when the output from a simulator cannot be parsed as a floating point
    number."""

    pass
