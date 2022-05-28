import subprocess
import typing

def ssh_run_command(host_machine: str, username: str, command: str) -> typing.Tuple[str]:
    """
    Run a command on host via ssh.

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
    Tuple:
        A tuple of two strings, the stdout and stderr in that order
    """
    ssh_command = ['ssh', username + '@' + host_machine, command]
    process = subprocess.Popen(ssh_command,
                     stdout = subprocess.PIPE, 
                     stderr = subprocess.PIPE,
                     text = True,
                     shell = False
                     )
    std_out, std_err = process.communicate()
    return std_out, std_err
    

def ssh_submit_job(host_machine: str, username: str, command: str) -> str:
    """
    Submit a job to a host via ssh. All jobs are submitted using the at scheduler.

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
    str:
        The id of submitted job on host machine
    """
    job_id = None
    run_delay = '2 minutes'
    submit_command = 'echo ' + '"' + command + '" | at now + ' + run_delay
    # The redirection below is because at appears to send job details to stderr
    # when started 
    submit_command = submit_command + ' 2>&1'
    std_out, std_err = ssh_run_command(host_machine=host_machine, username=username, command=submit_command) 
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
    str:
        Status of job
    """
    status = None
    check_status_command = 'atq'
    std_out, std_err = ssh_run_command(host_machine=host_machine, username=username, command=check_status_command)
    if std_err:
        print('ssh job status check failed with: ' + std_err)
        status = None
    else:
        atq_returned_lines = std_out.split('\n')
        for line in atq_returned_lines:
            if line != '':
                line_split = line.split()
                if line_split[0].strip() == job_id.strip():
                    status = line_split[6]
    return status