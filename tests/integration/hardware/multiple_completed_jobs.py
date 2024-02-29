import time
from typing import Any

from exauq.core.modelling import Input
from exauq.sim_management.hardware import JobStatus
from exauq.sim_management.jobs import Job, JobId
from tests.integration.hardware.utilities import (
    get_command_line_args,
    make_remote_server_script,
    read_json_config,
)


def run(ssh_config: dict[str, Any], remote_script_config: dict[str, Any]) -> None:
    # Create interface to remote server where we can run a script
    hardware = make_remote_server_script(ssh_config, remote_script_config)

    # Create jobs to submit
    jobs = [
        Job(id_=JobId(1), data=Input(1, 2, 3)),
        Job(id_=JobId(2), data=Input(4, 5, 6)),
    ]

    try:
        # Submit the jobs
        for job in jobs:
            hardware.submit_job(job)

        # Confirm that job status of each job is RUNNING.
        assert all(hardware.get_job_status(job.id) == JobStatus.RUNNING for job in jobs)

        # Wait for the jobs to complete
        time.sleep(4)

        # Check each job has completed
        assert all(hardware.get_job_status(job.id) == JobStatus.COMPLETED for job in jobs)

        # Check expected output value
        assert all(
            hardware.get_job_output(job.id) == float(sum(job.data)) for job in jobs
        )
    finally:
        # Clean up remote job directories
        for job in jobs:
            hardware.delete_remote_job_dir(job.id)
        pass


if __name__ == "__main__":
    args = get_command_line_args()
    ssh_config = read_json_config(args["ssh_config_path"])
    remote_script_config = read_json_config(args["remote_script_config_path"])
    run(ssh_config, remote_script_config)
