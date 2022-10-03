import subprocess


def local_install(install_command: str):
    """
    Install sim directory and files on local host

    Parameters
    ----------
    install_command: str
        the install command

    Returns
    -------
    Process:
        A subprocess popen object
    """
    local_install_command = ["bash", "-c", install_command]
    process = subprocess.Popen(
        local_install_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=False,
    )
    return process


def remote_install(install_command: str, host: str, user: str):
    """
    Install sim directory and files on local host

    Parameters
    ----------
    install_command: str
        the install command
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
        remote_install_command = ["ssh", user + "@" + host, install_command]
    else:
        remote_install_command = ["ssh", host, install_command]
    process = subprocess.Popen(
        remote_install_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=False,
    )
    return process


def local_run(command: str):
    """
    Run a command directly on local host.

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


def remote_run(command: str, host: str, user: str):
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
