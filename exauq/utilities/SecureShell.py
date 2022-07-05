import subprocess
import typing


def ssh_run(command: str, host: str, user: str) -> typing.Tuple[str]:
    """
    Run a command on host via ssh.

    Parameters
    ----------
    command: str
        command to run on host machine
    host: str
        host machine name
    user: str
        user name to run job on remote machine under

    Returns
    -------
    Tuple:
        A tuple of two strings, the stdout and stderr in that order
    """
    if user:
        ssh_command = ["ssh", user + "@" + host, command]
    else:
        ssh_command = ["ssh", host, command]
    process = subprocess.Popen(
        ssh_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=False,
    )
    return process
