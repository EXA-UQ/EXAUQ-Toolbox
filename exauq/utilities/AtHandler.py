from exauq.utilities.SecureShell import ssh_run
from exauq.utilities.JobStatus import JobStatus
from exauq.utilities.JobHandler import JobHandler

class AtHandler(JobHandler):
    """
     Class for handling jobs with the at scheduler
    """
    def __init__(self):
        self.handler_id = "at"

    def job_submit(command: str,  host_machine: str, username: str) -> str:
        """
        Method that submits a job via at and returns the job id

        Parameters
        ----------
        command: str
            command to run on host machine
        host_machine: str
            host machine name
        username: str
            username to run job on remote machine

        Returns
        -------
        str:
            the job id
        """
        submit_command = 'echo "' + command + ' 1> job.out 2> job.err' + '" | at now 2>&1'
        stdout, stderr = ssh_run(command=submit_command, host_machine=host_machine, username=username)
        if stderr:
            print('job submission failed with: ', stderr)
            job_id = None
        else:
            job_id = stdout.split()[1]
        return job_id


    def poll(job_id: str, host_machine: str, username: str) -> str:
        """
        Method that polls a job with atq given a job id and return status of the job

        Parameter
        ---------
        job_id: str
            the job id for which to poll
        host_machine: str
            host machine name
        username: str
            username  

        Returns
        -------
        str:
            the current status of the job
        """
        poll_command = 'atq'
        stdout, stderr = ssh_run(command=poll_command, host_machine=host_machine, username=username)
        if stderr:
            print('job polling failed with: ', stderr)
            job_status = None
        else:
            stdout_lines= [line for line in stdout.split("\n") if line]
            for line in stdout_lines:
                line_fields = line.split()
                if job_id.strip() == line_fields[0]:
                    at_status = line_fields[6] 
            if at_status == 'a':
                job_status = JobStatus.IN_QUEUE
            elif at_status == '=':
                job_status = JobStatus.RUNNING
            else:
                print("Status for job_id {} could not be acertained. \n output for atq: {}".format(job_id, stdout))
                job_status = None
        return job_status