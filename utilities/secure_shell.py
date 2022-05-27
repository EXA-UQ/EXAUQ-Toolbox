import subprocess
import typing

def ssh_submit_job(host_machine: str, username: str, command: str) -> str:
    """
    Submit a job to a host via ssh

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
    pass


def ssh_check_job_status(host_machine: str, username: str, command: str) -> str:
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
    pass