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
    local_command = ["bash", "-c", command]
    process = subprocess.Popen(
        local_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=False,
    )
    return process
