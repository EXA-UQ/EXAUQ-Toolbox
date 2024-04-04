import os

import daemon
from daemon import pidfile
from daemon_service import start_service


def run_daemon():
    # Define the path for the PID file to prevent multiple instances
    pid_file_path = "/tmp/exauq_daemon.pid"
    # Define the path for your Unix domain socket or other IPC mechanisms
    socket_path = "/tmp/exauq_daemon.sock"

    # Ensure the socket file does not exist before starting
    try:
        os.unlink(socket_path)
    except OSError:
        if os.path.exists(socket_path):
            raise

    # Daemon context setup
    context = daemon.DaemonContext(
        working_directory=os.getcwd(),  # Sets the current working directory
        umask=0o002,
        pidfile=pidfile.TimeoutPIDLockFile(pid_file_path),
    )

    log_directory = os.path.expanduser("~/exauq_logs/")
    if not os.path.exists(log_directory):
        os.makedirs(log_directory, exist_ok=True)

    context.stdout = open(os.path.join(log_directory, "exauq_daemon_stdout.log"), "w+")
    context.stderr = open(os.path.join(log_directory, "exauq_daemon_stderr.log"), "w+")

    with context:
        start_service(socket_path)


if __name__ == "__main__":
    run_daemon()
