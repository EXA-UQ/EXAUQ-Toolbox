import time
from exauq.utilities.SecureShell import ssh_run
from exauq.utilities.LocalRun import local_run
from exauq.utilities.JobStatus import JobStatus
from exauq.utilities.JobHandler import JobHandler


class BgHandler(JobHandler):
    """
    Class for submitting jobs as a background process
    """

    def submit_job(self, sim_id: str, command: str) -> None:
        """
        Method that runs a job as a background process using bash and returns the process id

        Parameters
        ----------
        sim_id: str
            id used to name stdout and stderr files - nominally should be set to simulator id.
        command: str
            command to run on host machine
        """
        if self.run_process is None:
            self.submit_time = time.strftime("%H:%M:%S", time.localtime())
            submit_command = "nohup bash -c '{0} || echo EXAUQ_JOB_FAILURE' > {1}.out 2> {1}.err & echo $!".format(
                command, sim_id
            )
            if self.run_local:
                self.run_process = local_run(command=submit_command)
            else:
                self.run_process = ssh_run(
                    command=submit_command, host=self.host, user=self.user
                )
            self.job_status = JobStatus.SUBMITTED

    def poll_job(self, sim_id: str) -> None:
        """
        Method that polls the process/job with the ps command and sets its status

        Parameter
        ---------
        sim_id: str
            id used to name stdout and stderr files - nominally would be set to simulator id.
        """
        self.last_poll_time = time.strftime("%H:%M:%S", time.localtime())
        if self.run_process is not None and self.job_id is None:
            if self.run_process.poll() is not None:
                stdout, stderr = self.run_process.communicate()
                if stderr:
                    print("job submission failed with: ", stderr)
                    self.job_status = JobStatus.SUBMIT_FAILED
                else:
                    self.job_id = stdout.split()[0]
                    self.job_status = JobStatus.RUNNING
                self.run_process = None
            return

        if self.poll_process is None and self.job_id is not None:
            poll_command = "ps aux {0}; tail -1 {1}.out".format(self.job_id, sim_id)
            if self.run_local:
                self.poll_process = local_run(command=poll_command)
            else:
                self.poll_process = ssh_run(
                    command=poll_command, host=self.host, user=self.user
                )
            return

        if self.poll_process is not None and self.poll_process.poll() is not None:
            stdout, stderr = self.poll_process.communicate()
            if stderr:
                print("job polling failed with: ", stderr)
            else:
                stdout_fields = stdout.split()
                if self.job_id.strip() in stdout_fields:
                    self.job_status = JobStatus.RUNNING
                elif "EXAUQ_JOB_FAILURE" in stdout_fields:
                    self.job_status = JobStatus.FAILED
                else:
                    self.job_status = JobStatus.SUCCESS
            self.poll_process = None
            return
