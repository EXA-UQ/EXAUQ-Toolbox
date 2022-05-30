import subprocess
import typing

def ssh_run(command: str, host: str, username: str) -> typing.Tuple[str]:
    """
    Run a command on host via ssh.

    Parameters
    ----------
    command: str
        command to run on host machine
    host: str
        host machine name
    username: str
        username to run job on remote machine

    Returns
    -------
    Tuple:
        A tuple of two strings, the stdout and stderr in that order
    """
    ssh_command = ['ssh', username + '@' + host, command]
    process = subprocess.Popen(ssh_command,
                     stdout = subprocess.PIPE, 
                     stderr = subprocess.PIPE,
                     text = True,
                     shell = False
                     )
    return process.communicate()