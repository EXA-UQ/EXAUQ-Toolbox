import subprocess
import typing
from urllib.parse import _NetlocResultMixinStr

def ssh_submit_job(host_machine: str, username: str, command: str) -> str:
    """
    Submit a job to a host via ssh. All jobs are submitted using the at scheduler.
    A default run delay of two minutes is hardcoded for now

    Parameters
    ----------
    host_machine: str
        host machine name
    username: str
        username to run job on remote machine
    command: str
        command to run on host machine

    Returns
    -------
    string:
        The id of submitted job on host machine
    """
    job_id = None

    run_delay = '2 minutes'
    submit_command = "'" + 'echo ' + '"' + command + '" | at now + ' + run_delay + "'"
    ssh_command = 'ssh ' + username + '@' + host_machine + ' ' + submit_command
    process = subprocess.Popen(ssh_command,
                     stdout = subprocess.PIPE, 
                     stderr = subprocess.PIPE,
                     text = True,
                     shell = True
                     )
    std_err, std_out = process.communicate() # There is something seriously odd going on here
                                             # the output from stdout seems to go into stderr.
    if std_err:
        print('ssh job submission failed with: ' + std_err)
        job_id = None
    else:
        at_returned_fields = std_out.split()
        job_id = at_returned_fields[1]
    return job_id


def ssh_check_job_status(host_machine: str, username: str, job_id: str) -> str:
    """
    Check status of job id running on host machine via ssh

    Parameters
    ----------
    host_machine: str
        host machine name
    username: str
        username to run job on remote machine
    job_id: str
        id of job to check
    
    Returns
    -------
    string:
        Status of job
    """
    status = None
    check_status_command = "'atq'"
    ssh_command = 'ssh ' + username + '@' + host_machine + ' ' + check_status_command
    process = subprocess.Popen(ssh_command,
                     stdout = subprocess.PIPE, 
                     stderr = subprocess.PIPE,
                     text = True,
                     shell = True
                     )
    std_out, std_err = process.communicate()
    if std_err:
        print('ssh job status check failed with: ' + std_err)
        status = None
    else:
        atq_returned_lines = std_out.split('\n')
        for line in atq_returned_lines:
            print(line)
            if line != '':
                line_split = line.split()
                if line_split[0].strip() == job_id.strip():
                    status = line_split[6]
    return status