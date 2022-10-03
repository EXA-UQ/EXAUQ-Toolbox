import time
from enum import Enum
from exauq.utilities.jobstatus import JobStatus
from exauq.utilities.paths import Paths
from exauq.utilities.comms import local_install, remote_install, local_run, remote_run


class SchedType(Enum):
    """
    Simulation status
    """

    BACKGROUND = 0
    AT = 1
    BATCH = 2


class JobHandler:
    """
    Class defining a job handler
    """

    def __init__(self, host: str, user: str, type: str, run_local=False) -> None:
        """
        Initialise a job handler object

        Parameters
        ----------
        host: str
            Name of host machine
        user: str
            Username of host account
        type: str
            Type of job scheduler to use when submitting job
        run_local: bool
            If True run job directly on current host machine, else run via remote login
        """

        self.host = host
        self.user = user
        self.run_local = run_local
        self.sim_dir = None
        self.install_process = None
        self.submit_process = None
        self.poll_process = None
        self.job_id = None
        self.job_status = None
        self.submit_time = None
        self.last_poll_time = None

        self.install_command = (
            """bash -c 'mkdir -p {0} ; touch {0}/std.out {0}/std.err'"""
        )

        if type == SchedType.BACKGROUND:
            self.type = type
            # Note the $! variable in bash returns the PID of the last command, which is what we want here
            self.run_command = """nohup bash -c '{0} || echo EXAUQ_JOB_FAILURE' > {1}/std.out 2> {1}/std.err & echo $!"""
            self.poll_command = """bash -c 'ps -p {0} -o pid= -o stat= -o comm= ; tail -1 {1}/std.out'"""
        if type == SchedType.AT:
            self.type = type
            # Note the command below uses the bash cut command to return the job id from the at scheduler output
            self.run_command = """echo "bash -c '{0} || echo EXAUQ_JOB_FAILURE' > {1}/std.out 2> {1}/std.err" | at now 2>&1 | cut -f2 -d ' '"""
            self.poll_command = """bash -c 'atq | grep {0} ; tail -1 {1}/std.out'"""
        if type == SchedType.BATCH:
            self.type = type
            # Note the command below uses the bash cut command to return the job id from the at scheduler output
            self.run_command = """echo "bash -c '{0} || echo EXAUQ_JOB_FAILURE' > {1}/std.out 2> {1}/std.err" | batch 2>&1 | cut -f2 -d ' '"""
            self.poll_command = """bash -c 'atq | grep {0} ; tail -1 {1}/std.out'"""

    def install_sim(self, sim_id: str) -> None:
        """
        Method that creates the necessary simulator job run directory and log files on the remote/local
        platform

        Parameters
        ----------
        sim_id: str
            The simulator id.
        """
        if self.install_process is None:
            self.sim_dir = Paths.SIM_RUN_DIR.format(sim_id)
            install_command = self.install_command.format(self.sim_dir)
            if self.run_local:
                self.install_process = local_install(install_command=install_command)
            else:
                self.install_process = remote_install(
                    install_command=install_command, host=self.host, user=self.user
                )

    def run_sim(self, command: str) -> None:
        """
        Method that submits a simulator job to a local or a remote scheduler

        Parameters
        ----------
        sim_id: str
            The simulator id.
        command: str
            Command to run on host machine
        """
        if self.submit_process is None:
            self.submit_time = time.strftime("%H:%M:%S", time.localtime())
            submit_command = self.run_command.format(command, self.sim_dir)
            if self.run_local:
                self.submit_process = local_run(command=submit_command)
            else:
                self.submit_process = remote_run(
                    command=submit_command, host=self.host, user=self.user
                )

    def poll_sim(self) -> None:
        """
        Method that polls the sim process/job and sets its status
        """

        self.last_poll_time = time.strftime("%H:%M:%S", time.localtime())

        # If an install process is active and the result from install process is
        # available then set sim status based on install process result and return
        if self.install_process is not None:
            if self.install_process.poll() is not None:
                stdout, stderr = self.install_process.communicate()
                if stderr:
                    print("sim installation failed with: ", stderr)
                    self.job_status = JobStatus.INSTALL_FAILED
                else:
                    self.job_status = JobStatus.INSTALLED
                self.install_process = None
            else:
                pass
            return

        # If a submit process is active and the result from submit process is
        # available then set sim status based on the submit process result and return.
        if self.submit_process is not None:
            if self.submit_process.poll() is not None:
                stdout, stderr = self.submit_process.communicate()
                if stderr:
                    print("job submission failed with: ", stderr)
                    self.job_status = JobStatus.SUBMIT_FAILED
                else:
                    self.job_id = stdout.split()[0]
                    self.job_status = JobStatus.SUBMITTED
                self.submit_process = None
            else:
                pass
            return

        # If a poll process is currently active and the poll process result is available,
        # then set the sim status based on the poll status results and return
        if self.poll_process is not None:
            if self.poll_process.poll() is not None:
                stdout, stderr = self.poll_process.communicate()
                if stderr:
                    print("job polling failed with: ", stderr)
                else:
                    stdout_fields = stdout.split()
                    if self.job_id in stdout_fields and self.job_id == stdout_fields[0]:
                        if self.type == SchedType.BACKGROUND:
                            self.job_status = JobStatus.RUNNING
                        if self.type == SchedType.AT:
                            if stdout_fields[-2] == "=":
                                self.job_status = JobStatus.RUNNING
                            if stdout_fields[-2] == "a":
                                self.job_status = JobStatus.IN_QUEUE
                        if self.type == SchedType.BATCH:
                            if stdout_fields[-2] == "=":
                                self.job_status = JobStatus.RUNNING
                            if stdout_fields[-2] == "b":
                                self.job_status = JobStatus.IN_QUEUE
                    elif "EXAUQ_JOB_FAILURE" in stdout_fields:
                        self.job_status = JobStatus.FAILED
                    else:
                        self.job_status = JobStatus.SUCCESS
                self.poll_process = None
            else:
                pass
            return

        # If a poll process is currently not active and a job id is available, then start
        # a new poll process and return
        if self.poll_process is None:
            if self.job_id is not None:
                poll_command = self.poll_command.format(self.job_id, self.sim_dir)
                if self.run_local:
                    self.poll_process = local_run(command=poll_command)
                else:
                    self.poll_process = remote_run(
                        command=poll_command, host=self.host, user=self.user
                    )
            else:
                pass
            return
