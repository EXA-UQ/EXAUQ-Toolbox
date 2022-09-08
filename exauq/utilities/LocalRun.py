import subprocess


def local_run(command: str):
    """
    Run a command on host via ssh.

    Parameters
    ----------
    command: str
        command to run on host machine

    Returns
    -------
    Process:
        A subprocess popen object
    """
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=False,
    )
    return process
