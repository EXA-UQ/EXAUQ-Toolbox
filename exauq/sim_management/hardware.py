import getpass
import io
import pathlib
import string
import textwrap
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional, Union

from fabric import Config, Connection
from paramiko.ssh_exception import AuthenticationException, SSHException

from exauq.sim_management.jobs import Job, JobId
from exauq.sim_management.types import FilePath


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

    PENDING_SUBMIT = "Pending submit"
    """A job has been set up locally but has not yet been submitted to a remote
    machine. Has the value 'Pending submit'."""

    FAILED_SUBMIT = "Failed submit"
    """A job has been set up locally however submission to remote machine failed.
    Has the value 'Failed submit'."""

    CANCELLED = "Cancelled"
    """A job has been cancelled from a locally issued request or intervention. Has the
    value 'Cancelled'."""

    PENDING_CANCEL = "Pending cancel"
    """A job has been requested to be cancelled but has not yet been cancelled. Has the 
    value 'Pending cancel'."""

    FAILED_CANCEL = "Failed cancel"
    """A job has been requested to be cancelled but the cancellation failed. Has the
    value 'Failed cancel'."""


TERMINAL_STATUSES = {
    JobStatus.COMPLETED,
    JobStatus.FAILED,
    JobStatus.FAILED_SUBMIT,
    JobStatus.CANCELLED,
}
"""Statuses that define jobs that are no longer running (possibly due to error)."""

PENDING_STATUSES = {
    JobStatus.PENDING_CANCEL,
    JobStatus.PENDING_SUBMIT,
}
"""Statuses that define jobs that are waiting to be run or cancelled."""


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
    def __init__(self):
        self._level = 0
        self._tag = None

    @property
    def level(self) -> int:
        """(Read-only) The level of the hardware interface."""
        return self._level

    @property
    def tag(self) -> Optional[str]:
        """(Read-only) The tag of the hardware interface."""
        return self._tag

    @level.setter
    def level(self, value: int):
        if not isinstance(value, int):
            raise TypeError(f"Expected level ({value}) to be of type int. Got {type(value)} instead.")

        self._level = value

    @tag.setter
    def tag(self, value: str):
        if not isinstance(value, str):
            raise TypeError(f"Expected tag ({value}) to be of type str. Got {type(value)} instead.")

        self._tag = value

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
    key_filename : exauq.sim_management.types.FilePath, optional
        The path to the SSH private key file to authenticate with the SSH server.
    ssh_config_path : exauq.sim_management.types.FilePath, optional
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
        key_filename: Optional[FilePath] = None,
        ssh_config_path: Optional[FilePath] = None,
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
                f"{user}@{host}", connect_kwargs={"key_filename": str(key_filename)}
            )
        elif ssh_config_path is not None:
            ssh_config = Config(overrides={"ssh_config_path": str(ssh_config_path)})
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


class _Template(string.Template):
    """Subclass of ``string.Template`` that changes the default delimiter.

    Text that begins with '#PY_' will be replaced when applying text substitution.

    Examples
    --------

    >>> _Template("something=#PY_FOO").substitute({"FOO": "foo"})
    "something=foo"
    """

    delimiter = "#PY_"


class UnixServerScriptInterface(SSHInterface):
    """Interface for running a simulation script on a Unix server over SSH.

    This interface is designed to invoke simulation code on a remote server with the
    following command:

    ```
    <program> <script_path> path/to/simulation-input.csv path/to/simulation-output.txt
    ```

    Here, ``<program>`` is a program that can run the simulator script, such as ``bash``,
    ``python``, ``Rscript``, etc. The first argument to the script is the path to a CSV
    file containing the simulation input data, while the second argument is the path to a
    text file that the script will write the simulation output to.

    Objects of this class will take care of creating the input CSV file from a `Job`,
    uploading this to a job-specific subdirectory of a remote 'workspace' directory. This
    job-specific directory is also where the output of the simulator script is written to,
    along with a file capturing standard output and standard error from running the
    script. Note that any required intermediary directories are created on the server.

    The remote workspace directory can be specified as part of initialisation. If a a
    pre-existing directory is supplied, then the details of any existing jobs already
    recorded in the workspace directory will be retrieved and cached. If a workspace
    directory is not specified, then a new directory will be created alongside the main
    simulation script with name'exauqXXXXX' where 'XXXXX' is a uniquely generated string
    of characters created via the ``mktemp`` command on Unix systems.

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
    script_path : exauq.sim_management.types.FilePath
        The path to the script on the Unix server to run with `program`.
    workspace_dir : exauq.sim_management.types.FilePath, optional
        (Default: None) A path to a directory on the Unix server where job-specific
        subdirectories should be created. Relative paths will be relative to the default
        working directory for a new SSH session (usually the user's home directory). If
        ``None`` then a new directory will be created alongside the script defined by
        `script_path`.
    key_filename : exauq.sim_management.types.FilePath, optional
        (Default: None) The path to an SSH private key file to authenticate with the SSH
        server. The key file must be unencrypted.
    ssh_config_path : exauq.sim_management.types.FilePath, optional
        (Default: None) The path to an SSH configuration file.
    use_ssh_agent : bool, optional
        (Default: False) If ``True``, use a running SSH agent for authentication.
    max_attempts : int, optional
        (Default: 3) The number of authentication attempts allowed.

    Attributes
    ----------
    workspace_dir : str, optional
        (Read-only) The directory within which details of jobs are recorded, or None
        if this is unknown at the time of calling.

    Raises
    ------
    ValueError
        If more than one method of authentication is provided.
    HardwareInterfaceFailureError:
        If there were problems connecting to the server, establishing the existence of
        the workspace directory, or other such server-related issues.
    """

    _bash = "/bin/bash"
    _manager_script_name = "exauq_manager.sh"
    _runner_script_name = "runner.sh"

    def __init__(
        self,
        user: str,
        host: str,
        program: str,
        script_path: FilePath,
        workspace_dir: Optional[FilePath] = None,
        key_filename: Optional[FilePath] = None,
        ssh_config_path: Optional[FilePath] = None,
        use_ssh_agent: Optional[bool] = False,
        max_attempts: int = 3,
    ):
        super().__init__(
            user, host, key_filename, ssh_config_path, use_ssh_agent, max_attempts
        )
        self._user = user
        self._host = host
        self._user_at_host = f"{self._user}@{self._host}"
        self._program = program
        self._script_path = pathlib.PurePosixPath(script_path)
        self._workspace_dir = (
            pathlib.PurePosixPath(workspace_dir) if workspace_dir is not None else None
        )
        if not self._remote_dir_exists(workspace_dir):
            self._make_workspace_dir()
            self._job_log = dict()
        else:
            self._job_log = self._initialise_job_log_from_server()

    def _remote_dir_exists(self, path: Union[str, pathlib.PurePosixPath, None]) -> bool:
        """Whether a directory at the given path exists on the server."""

        if path is None:
            return False
        else:
            flag = "EXISTS"
            try:
                result = self._run_remote_command(
                    f"if [ -d {path} ]; then echo {flag}; fi"
                )
            except Exception as e:
                raise HardwareInterfaceFailureError(
                    f"Could not establish existence of workspace directory {self.workspace_dir} "
                    f"for {self._user_at_host}: {e}"
                )
            return flag == result

    def _initialise_job_log_from_server(self) -> dict[str, dict[str, Any]]:
        """Populate the job log with details of existing jobs that have been submitted to
        the server."""

        # Check whether there are any jobs already submitted
        job_ids = self._fetch_remote_job_ids()

        return {
            job_id: self._make_job_settings(job_id, status=JobStatus.SUBMITTED)
            for job_id in job_ids
        }

    def _fetch_remote_job_ids(self) -> tuple[JobId, ...]:
        """Get IDs of jobs that have been submitted to the server.

        Gathers the names of directories directly below the workspace directory that
        have integer names and contain a manager script.
        """
        # List paths to job manager scripts in directories directly below the workspace
        # directory
        no_job_ids_flag = "NO_JOBIDS"
        cmd = f"cd {self.workspace_dir} && find . | grep -G '^\\./[0-9]*/{self._manager_script_name}$' || echo {no_job_ids_flag}"
        try:
            job_manager_paths_str = self._run_remote_command(cmd)
        except Exception as e:
            raise HardwareInterfaceFailureError(
                f"Could not fetch job IDs from workspace directory {self.workspace_dir} "
                f"for {self._user_at_host}: {e}"
            )

        if not job_manager_paths_str == no_job_ids_flag:
            # Extract the job IDs as names of directories containing the job manager scripts
            job_manager_paths = [
                pathlib.PurePosixPath(path) for path in job_manager_paths_str.split("\n")
            ]
            return tuple(JobId(path.parent.name) for path in job_manager_paths)
        else:
            return tuple()

    def _make_job_settings(
        self, job_id: JobId, status: JobStatus = JobStatus.PENDING_SUBMIT
    ) -> dict[str, Any]:
        """Make settings for specifying a job on the server.

        It is expected that the resulting dict will constitute an entry in the job log.
        """

        job_remote_dir = self._workspace_dir / str(job_id)
        job_manager_path = job_remote_dir / self._manager_script_name
        runner_path = job_remote_dir / self._runner_script_name
        input_data_path = pathlib.PurePosixPath(job_remote_dir, "input.csv")
        script_output_path = str(job_remote_dir / "output.txt")
        script_stdout_path = job_remote_dir / f"{self._script_path.name}.out"
        return {
            "status": status,
            "job_remote_dir": job_remote_dir,
            "runner": runner_path,
            "job_manager": job_manager_path,
            "input_data_path": input_data_path,
            "script_output_path": script_output_path,
            "script_stdout_path": script_stdout_path,
            "output": None,
        }

    @property
    def workspace_dir(self) -> Optional[str]:
        """(Read-only) The directory within which details of jobs are recorded, or None
        if this is unknown at the time of calling (e.g. because it hasn't been created
        yet)."""

        return str(self._workspace_dir) if self._workspace_dir is not None else None

    def submit_job(self, job: Job, resubmit: bool = False) -> None:
        """Submit a job for the simulation code.

        Upon submission, a new subdirectory of the remote workspace directory is created
        for the job, using the job's ID as the directory name. (The workspace directory
        will be created as well if it doesn't already exist.) A CSV file containing the
        simulation input data for the job is uploaded to this directory.

        A Bash script is also uploaded to the job's directory, which is responsible for
        managing the job; it is through this script that the job can be started, cancelled
        or its status retrieved.

        If a job with the same ID has already been submitted to the server and `resubmit`
        is ``False``, then an error will be raised. Only jobs that have completed, be it
        successfully or ending in failure, or that have been cancelled, may be
        resubmitted. In this case, calling with `resubmit` set to ``True`` will delete any
        existing remote-side data for the corresponding job directory and then submit the
        supplied `job`.

        Parameters
        ----------
        job : Job
            A job containing the data to run the simulation code with.
        resubmit : bool
            (Default: False) Whether the job is being resubmitted, i.e. whether to delete
            any existing remote-side data for job before submission.

        Raises
        ------
        ValueError
            If a job with the same ID has already been submitted and ``resubmit = False``,
            or if ``resubmit = True`` and a job with the same ID has not completed.
        HardwareInterfaceFailure
            If there were problems connecting to the server, making files / directories on
            the server or other such server-related problems.
        """

        if resubmit:
            # Clear the artifacts from the remote side, as long as the job is not running
            # or marked as submitted.
            if (status := self.get_job_status(job.id)) not in {
                JobStatus.SUBMITTED,
                JobStatus.RUNNING,
            }:
                self.delete_remote_job_dir(job.id)
                del self._job_log[job.id]
            else:
                raise ValueError(
                    f"Cannot resubmit job with ID {job.id} as job status is '{status.value}'. "
                    "(Cancel the job before resubmitting.)"
                )

        elif self._job_has_been_submitted(job.id):
            raise ValueError(
                f"Cannot submit job with ID {job.id}: a job with the same ID has already "
                f"been submitted."
            )

        # Create the settings for the new job
        job_settings = self._make_job_settings(job.id)

        # Make job-specific remote workspace directory (will raise error if directory
        # already exists)
        self._make_directory_on_remote(job_settings["job_remote_dir"])

        # Put simulator input data onto server
        data_str = ",".join(map(str, job.data)) + "\n"
        self._make_text_file_on_remote(data_str, job_settings["input_data_path"])

        # Create runner script and manager script and put onto server
        runner_script = self._make_runner_script(
            job_settings["job_remote_dir"],
            job_settings["input_data_path"],
            job_settings["script_output_path"],
        )
        manager_script = self._make_manager_script(
            job_settings["job_remote_dir"],
            job_settings["runner"],
            job_settings["script_stdout_path"],
            job_settings["script_output_path"],
        )
        self._make_text_file_on_remote(runner_script, job_settings["runner"])
        self._make_text_file_on_remote(manager_script, job_settings["job_manager"])

        # Start job
        try:
            _ = self._run_remote_command(
                f"{self._bash} {job_settings['job_manager']} start"
            )
        except Exception as e:
            raise HardwareInterfaceFailureError(
                f"Could not start job with id {job.id} on {self._user_at_host}: {e}"
            )

        # Mark job as submitted and store settings in job log
        job_settings["status"] = JobStatus.SUBMITTED
        self._job_log[job.id] = job_settings

        return None

    def _make_workspace_dir(self):
        """Make the server-side workspace directory.

        If the path to a directory was provided explicitly during object initialisation,
        then create that directory on the server. Otherwise, create a default directory
        alongside the simulator script. The name of the directory will be of the form
        'exauqXXXXX' where 'XXXXX' is a uniquely generated string of characters created
        via the ``mktemp`` command on Unix systems."""

        if self.workspace_dir is None:
            try:
                workspace_dir_str = self._run_remote_command(
                    f"mktemp -d -p {self._script_path.parent} exauqXXXXX"
                )
                self._workspace_dir = pathlib.PurePosixPath(workspace_dir_str)
            except Exception as e:
                raise HardwareInterfaceFailureError(
                    f"Could not create workspace directory in {self._script_path.parent} "
                    f"for {self._user_at_host}: {e}"
                )
            return None
        else:
            self._make_directory_on_remote(self._workspace_dir, make_parents=True)
            return None

    def _make_runner_script(
        self,
        job_remote_dir: Union[str, pathlib.PurePosixPath],
        input_path: Union[str, pathlib.PurePosixPath],
        output_path: Union[str, pathlib.PurePosixPath],
    ) -> str:
        """Create the text for a script that runs the simulation script."""
        template_str = r"""
        #!/bin/bash

        # Run script and create new COMPLETED flag file upon successful execution and
        # presence of output file.
        #PY_PROGRAM #PY_SCRIPT #PY_INPUT_PATH #PY_OUTPUT_PATH && if [ -e #PY_OUTPUT_PATH ]; then touch #PY_JOB_DIR/COMPLETED; fi

        """

        template_str = template_str[1:]  # remove leading newline character
        template = _Template(textwrap.dedent(template_str))
        return template.substitute(
            {
                "JOB_DIR": str(job_remote_dir),
                "SCRIPT": str(self._script_path),
                "PROGRAM": str(self._program),
                "INPUT_PATH": str(input_path),
                "OUTPUT_PATH": str(output_path),
            }
        )

    def _make_manager_script(
        self,
        job_remote_dir: Union[str, pathlib.PurePosixPath],
        runner_path: Union[str, pathlib.PurePosixPath],
        stdout_path: Union[str, pathlib.PurePosixPath],
        output_path: Union[str, pathlib.PurePosixPath],
    ) -> str:
        """Create the text for the server-side job management Bash program."""

        template_str = r"""
        #!/bin/bash

        # This script provides an interface for working with processes -- a kind of
        # very basic 'job' manager (where 'job' means a process and collection of
        # subprocesses, not a 'job' as would be worked with using e.g. the jobs program.)
        #
        # Arg 1: One of: start, status, stop.

        job_dir=#PY_JOB_DIR
        runner=#PY_RUNNER
        script_stout_sterr=#PY_STDOUT_PATH
        script_output=#PY_OUTPUT_PATH
        pid_file="${job_dir}/PID"
        pgid_file="${job_dir}/PGID"
        jobid_file="${job_dir}/JOBID"
        stopped_flag_file="${job_dir}/STOPPED"
        completed_flag_file="${job_dir}/COMPLETED"
        failed_flag_file="${job_dir}/FAILED"

        FAILED_JOBID=",,"

        # Print an error message and exit with nonzero status.
        # Arg 1: String containing details of the error.
        error() {
            echo -e "${0}: error: ${1}" >&2
            exit 1
        }

        # Check that the current shell is set up in the required way for this script.
        check_system() {
            if ! [[ "$SHELL" =~ /bash$ ]]
            then
                error "must be running in a Bash shell"
            elif ! which pkill > /dev/null 2>&1
            then
                error "required command 'pkill' not available on system"
            elif ! which mktemp > /dev/null 2>&1
            then
                error "required command 'mktemp' not available on system"
            fi
        }

        # Run a job in the background and capture the PID of the process
        run_job() {
            nohup /bin/bash "$runner" >& "$script_stout_sterr" < /dev/null &
            runner_pid=$!
            echo $runner_pid | tr -d '[:space:]' > $pid_file
            ps -p $runner_pid -o pgid= | tr -d '[:space:]' > $pgid_file
        }

        # Get a unique identifier for a process, utilising the PID, start time (long
        # format) and user.
        # Arg1: a pid
        get_process_identifier() {
            echo "$(ps -p "${1}" -o user=),$(ps -p "${1}" -o pid=),$(ps -p "${1}" -o lstart=)"
        }

        PENDING_SUBMIT="PENDING_SUBMIT"
        RUNNING="RUNNING"
        STOPPED="STOPPED"
        COMPLETED="COMPLETED"
        FAILED="FAILED"

        record() {
            case $1 in
            "$STOPPED")
                touch $stopped_flag_file;;
            "$COMPLETED")
                touch $completed_flag_file;;
            "$FAILED")
                touch $failed_flag_file;;
            *)
                error "in function record: unsupported arg '${1}'";;
            esac
        }

        # Start the job and capture an ID for it.
        start_job() {
            run_job
            job_pid=$(cat $pid_file)
            jobid=$(get_process_identifier "${job_pid}")

            # If identifier is empty this implies the process is no-longer running,
            # which almost certainly suggests there was an error.
            if [ "$jobid" = "$FAILED_JOBID" ] && [ ! -e "$script_output" ]
            then
                record $FAILED
                error "script failed to run:\n$(cat ${script_stout_sterr})"
            fi
            echo "$jobid" > $jobid_file
        }


        # Get the status of a job
        get_status() {
            if [ ! -e $pid_file ]
            then
                echo $PENDING_SUBMIT
            elif [ -e $stopped_flag_file ]
            then
                echo $STOPPED
            elif [ -e $completed_flag_file ]
            then
                echo $COMPLETED
            elif [ -e $failed_flag_file ] || [ ! -e $jobid_file ]
            then
                echo $FAILED
            else
                # If here then the job was last known to be running
                job_id=$(cat $jobid_file)
                job_pid=$(cat $pid_file)
                current_id=$(get_process_identifier "${job_pid}")
                if [ ! "$job_id" = "$FAILED_JOBID" ] && [ "$job_id" = "$current_id" ]
                then
                    echo $RUNNING
                elif [ -e "$script_output" ]
                then
                    # For a job to be completed, it must not be running and have an
                    # output file.
                    record $COMPLETED
                    echo $COMPLETED
                else
                    record $FAILED
                    echo $FAILED
                fi
            fi
        }

        # Stop (cancel) a job by killing all processes within its group.
        stop_job() {
            status=$(get_status)
            if [ "$status" = "$RUNNING" ]
            then
                if xargs pkill -g < $pgid_file
                then
                    record $STOPPED
                fi
            fi
        }

        # Dispatch on command line arg
        check_system
        case $1 in
        start)
            start_job;;
        stop)
            stop_job;;
        status)
            get_status;;
        *)
            error "unsupported arg '${1}'";;
        esac

        """

        template_str = template_str[1:]  # remove leading newline character
        template = _Template(textwrap.dedent(template_str))
        return template.substitute(
            {
                "JOB_DIR": str(job_remote_dir),
                "RUNNER": str(runner_path),
                "STDOUT_PATH": str(stdout_path),
                "OUTPUT_PATH": str(output_path),
            }
        )

    def _make_directory_on_remote(
        self, path: Union[str, pathlib.PurePosixPath], make_parents: bool = False
    ) -> None:
        """Make a directory at the given path on the remote machine.

        If the directory already exists, then this will be left untouched without error if
        `make_parents` is ``True``, whereas an error will be raised if `make_parents` is
        ``False``. If `make_parents` is ``True`` then intermediary directories will be
        created as required (by calling ``mkdir`` with the ``-p`` option).
        """

        mkdir_command = (
            f"mkdir {path}" if not make_parents else f"[ -d {path} ] || mkdir -p {path}"
        )
        try:
            _ = self._run_remote_command(mkdir_command)
        except Exception as e:
            raise HardwareInterfaceFailureError(
                f"Could not make directory {path} for {self._user_at_host}: {e}"
            )
        return None

    def _make_text_file_on_remote(
        self, file_contents: str, target_path: Union[str, pathlib.PurePosixPath]
    ) -> None:
        """Make a text file on the remote machine with a given string as contents."""

        try:
            _ = self._conn.put(
                io.StringIO(file_contents),
                remote=str(target_path),
            )
        except Exception as e:
            raise HardwareInterfaceFailureError(
                f"Could not create text file at {target_path} for "
                f"{self._user_at_host}: {e}"
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
        `JobStatus.PENDING_SUBMIT`.

        A job that has successfully been started on the server will have a status of
        `JobStatus.RUNNING` (which, in this case, is equivalent to `JobStatus.SUBMITTED`).
        The status will remain as `JobStatus.RUNNING` until the corresponding remote
        process has stopped, at which point the status is determined as follows:

        * If an output file from the simulator has been created, then the status is
          ``JobStatus.COMPLETED``.
        * If the job was cancelled before completion, then the status is
          ``JobStatus.CANCELLED`.
        * If the job was not cancelled but no output file was created, then the status
          is `JobStatus.FAILED`. In particular, note that the exit code of the
          simulator script is not taken into account when determining whether a job has
          finished successfully or not.

        Parameters
        ----------
        job_id : JobId
            The ID of the job to check the status of.

        Returns
        -------
        JobStatus
            The status of the job.

        Raises
        ------
        HardwareInterfaceFailure
            If there were problems connecting to the server or retrieving the status of
            the job.
        """
        if not self._job_has_been_submitted(job_id):
            return JobStatus.PENDING_SUBMIT
        elif self._job_log[job_id]["status"] in {JobStatus.RUNNING, JobStatus.SUBMITTED}:
            self._update_status_from_remote(job_id)
            return self._job_log[job_id]["status"]
        else:
            return self._job_log[job_id]["status"]

    def _job_has_been_submitted(self, job_id: JobId) -> bool:
        """Whether a job with the given ID has been submitted."""

        return job_id in self._job_log

    def _update_status_from_remote(self, job_id: JobId) -> None:
        """Update the status of a job based on the status of the corresponding process on
        the server."""

        status = self._run_remote_command(
            f"{self._bash} {self._job_log[job_id]['job_manager']} status"
        )
        if status == "RUNNING":
            self._job_log[job_id]["status"] = JobStatus.RUNNING
        elif status == "COMPLETED":
            self._job_log[job_id]["status"] = JobStatus.COMPLETED
        elif status == "STOPPED":
            self._job_log[job_id]["status"] = JobStatus.CANCELLED
        else:
            self._job_log[job_id]["status"] = JobStatus.FAILED

        return None

    def _remote_job_is_running(self, job_id: JobId) -> bool:
        """Whether the remote process of a given job is running."""

        return self.get_job_status(job_id) == JobStatus.RUNNING

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

        elif self.get_job_status(job_id) == JobStatus.COMPLETED:
            output_path = self._job_log[job_id]["script_output_path"]
            output = self._retrieve_output(output_path)
            try:
                self._job_log[job_id]["output"] = (
                    float(output) if output is not None else output
                )
            except ValueError:
                raise SimulatorOutputParsingError(
                    f"Could not parse simulator output {output} for job ID {job_id} as a "
                    "float."
                )
            return self._job_log[job_id]["output"]
        else:
            return None

    def _retrieve_output(
        self, remote_path: Union[str, pathlib.PurePosixPath]
    ) -> Optional[str]:
        """Get the output of a simulation from the remote server."""

        with io.BytesIO() as buffer:
            try:
                _ = self._conn.get(str(remote_path), local=buffer)
            except FileNotFoundError:
                return None
            except Exception as e:
                raise HardwareInterfaceFailureError(
                    f"Could not retrieve output of script {self._script_path} from file "
                    f"{remote_path}: {e}"
                )

            contents = buffer.getvalue().decode(encoding="utf-8")

        return contents.strip()

    def cancel_job(self, job_id: JobId) -> None:
        """Cancel the job with a given job ID.

        Cancelling a job involves terminating the server-side simulator script process
        (and any subprocesses) associated with the job. If the job is not running (i.e.
        has completed, has failed or has already been cancelled) then this method will
        return without error.

        Parameters
        ----------
        job_id : JobId
            The ID of the job to cancel.

        Raises
        ------
        HardwareInterfaceFailureError
            If there were problems connecting to the server or otherwise cancelling the
            job.
        """
        if self.get_job_status(job_id) == JobStatus.RUNNING:
            try:
                self._run_remote_command(
                    f"{self._bash} {self._job_log[job_id]['job_manager']} stop"
                )
                self._job_log[job_id]["status"] = JobStatus.CANCELLED
            except Exception as e:
                raise HardwareInterfaceFailureError(
                    f"Could not cancel job with id {job_id}: {e}"
                )
            return None
        else:
            return None

    def delete_workspace(self) -> None:
        """Delete the entire workspace directory associated with this instance.

        Warning: this is an 'unsafe' deletion: it does not wait for any outstanding jobs
        to complete. This could result in server-side errors for any simulations that are
        still running when the workspace directory is deleted.

        Raises
        ------
        HardwareInterfaceFailureError
            If there were problems connecting to the server or deleting the directory.
        """
        try:
            _ = self._run_remote_command(f"rm -r {self.workspace_dir}")
        except Exception as e:
            raise HardwareInterfaceFailureError(
                f"Could not delete workspace directory {self.workspace_dir} for "
                f"{self._user_at_host}: {e}"
            )
        return None

    def delete_remote_job_dir(self, job_id: JobId) -> None:
        """Delete the remote directory corresponding to a given job ID.

        This will recursively delete all the contents of the directory, invoking
        ``rm -r`` on it. Only submitted jobs that aren't currently running can have their
        remote directories deleted.

        Parameters
        ----------
        job_id : JobId
            The ID of the job whose remote directory should be deleted.

        Raises
        ------
        ValueError
            If the supplied job ID has not been submitted, or if the job is still running.
        HardwareInterfaceFailure
            If there were problems connecting to the server or deleting the directory.
        """

        job_status = self.get_job_status(job_id)
        if job_status == JobStatus.PENDING_SUBMIT:
            raise ValueError(
                f"Cannot delete directory for job ID {job_id}: job has not been submitted."
            )

        elif job_status == JobStatus.RUNNING:
            raise ValueError(
                f"Cannot delete directory {self._job_log[job_id]['job_remote_dir']} for "
                f"job ID {job_id}: job is currently running."
            )
        else:
            job_remote_dir = self._job_log[job_id]["job_remote_dir"]
            deletion_cmd = f"rm -r {job_remote_dir}"
            try:
                _ = self._run_remote_command(deletion_cmd)
            except Exception as e:
                raise HardwareInterfaceFailureError(
                    f"Could not delete remote folder {job_remote_dir} for "
                    f"{self._user_at_host}: {e}"
                )

            return None


class HardwareInterfaceFailureError(Exception):
    """Raised when an error was encountered when running a command or communicating with a
    machine."""

    pass


class SimulatorOutputParsingError(Exception):
    """Raised when the output from a simulator cannot be parsed as a floating point
    number."""

    pass
