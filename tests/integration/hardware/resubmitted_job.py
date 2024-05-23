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
    hardware = make_unix_server_script_interface(ssh_config, remote_script_config)

    # Create a job to submit
    job = Job(id_=JobId(1), data=Input(1, 2, 3))

    try:
        # Submit the job
        hardware.submit_job(job)

        # Check workspace directory is not None
        assert hardware.workspace_dir is not None

        # Cancel the job
        hardware.cancel_job(job.id)

        # Check job has CANCELLED status
        assert hardware.get_job_status(job.id) == JobStatus.CANCELLED

        # Resubmit the job
        hardware.submit_job(job, resubmit=True)

        # Check job is running
        assert hardware.get_job_status(job.id) == JobStatus.RUNNING

        # Wait for job to complete
        time.sleep(3)

        # Check job is completed
        assert hardware.get_job_status(job.id) == JobStatus.COMPLETED
    finally:
        # Clean up workspace
        time.sleep(3)  # wait for job to complete
        hardware.delete_workspace()
        pass


if __name__ == "__main__":
    args = get_command_line_args()
    ssh_config = read_json_config(args["ssh_config_path"])
    remote_script_config = read_json_config(args["remote_script_config_path"])
    run(ssh_config, remote_script_config)
