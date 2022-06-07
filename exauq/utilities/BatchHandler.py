from exauq.utilities.SecureShell import ssh_run
from exauq.utilities.JobStatus import JobStatus
from exauq.utilities.JobHandler import JobHandler

class BatchHandler(JobHandler):
    """
     Class for handling jobs with the batch scheduler
    """
    def submit_job(self, sim_id: str, command: str) -> str:
        """
        Method that submits a job via batch and returns the job id

        Parameters
        ----------
        sim_id: str
            id used to name stdout and stderr files - nominally should be set to simulator id.
        command: str
            command to run on host machine

        Returns
        -------
        str:
            the job id
        """
        redirect_com = '1> {0}.out 2> {0}.err'.format(sim_id)
        submit_command = 'echo "(' + command + ' || echo EXAUQ_JOB_FAILURE)' + redirect_com + '" | batch 2>&1'
        stdout, stderr = ssh_run(command=submit_command, host=self.host, user=self.user)
        if stderr:
            print('job submission failed with: ', stderr)
            job_id = None
        else:
            job_id = stdout.split()[1]
        return job_id


    def poll_job(self, sim_id: str, job_id: str) -> JobStatus:
        """
        Method that polls a job with atq given a job id and return status of the job

        Parameter
        ---------
        sim_id: str
            id used to name stdout and stderr files - nominally would be set to simulator id.
        job_id: str
            the job id for which to poll

        Returns
        -------
        JobStatus:
            the current status of the job
        """
        poll_command = 'atq; tail -1 {0}.out'.format(sim_id) 
        stdout, stderr = ssh_run(command=poll_command, host=self.host, user=self.user)

        job_status = None
        if stderr:
            print('job polling failed with: ', stderr)
        else:
            stdout_fields = stdout.split()
            if job_id.strip() in stdout_fields:
                if job_id.strip() == stdout_fields[0] and stdout_fields[6] == '=':
                    job_status = JobStatus.RUNNING
                if job_id.strip() == stdout_fields[0] and stdout_fields[6] == 'b':
                    job_status = JobStatus.IN_QUEUE
            elif "EXAUQ_JOB_FAILURE" in stdout_fields:
                    job_status = JobStatus.FAILED
            else:
                job_status = JobStatus.SUCCESS
            if job_status is None:
                print("Status for job_id {} could not be acertained. \n output for polling command: {}".format(job_id, stdout))
        return job_status