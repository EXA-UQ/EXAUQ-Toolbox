import subprocess


def ssh_run(command: str, host: str, user: str):
    """
    Run a command on remote machine via ssh.

    Parameters
    ----------
    command: str
        command to run on remote machine
    host: str
        host machine name
    user: str
        user name to run job on remote machine under

    Returns
    -------
    Process:
        A subprocess popen object
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
