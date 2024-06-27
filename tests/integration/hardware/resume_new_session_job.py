import time
from typing import Any

from exauq.core.modelling import Input
from exauq.sim_management.hardware import JobStatus
from exauq.sim_management.jobs import Job, JobId
from tests.integration.hardware.utilities import (
    get_command_line_args,
    make_unix_server_script_interface,
    read_json_config,
)


def run(ssh_config: dict[str, Any], remote_script_config: dict[str, Any]) -> None:
    # Create interface to remote server where we can run a script
    hardware1 = make_unix_server_script_interface(ssh_config, remote_script_config)

    # Create jobs to submit
    job1 = Job(id_=JobId(1), data=Input(1, 2, 3))
    job2 = Job(id_=JobId(2), data=Input(0.1, 0.2, 0.3))

    try:
        # Submit the jobs
        hardware1.submit_job(job1)
        hardware1.submit_job(job2)

        # Check workspace directory is not None
        assert hardware1.workspace_dir is not None

        # Create new interface to remote server, using the same workspace directory as
        # the original interface.
        hardware2 = make_unix_server_script_interface(
            ssh_config, remote_script_config, workspace_dir=hardware1.workspace_dir
        )

        # Check that the job statuses are not PENDING_SUBMIT
        assert (not hardware2.get_job_status(job1.id) == JobStatus.PENDING_SUBMIT) and (
            not hardware2.get_job_status(job2.id) == JobStatus.PENDING_SUBMIT
        )

    finally:
        # Clean up remote workspace directory
        time.sleep(4)  # wait for jobs to complete
        hardware1.delete_workspace()
        pass


if __name__ == "__main__":
    args = get_command_line_args()
    ssh_config = read_json_config(args["ssh_config_path"])
    remote_script_config = read_json_config(args["remote_script_config_path"])
    run(ssh_config, remote_script_config)
