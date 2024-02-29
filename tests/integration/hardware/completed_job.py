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

    # Create a job to submit
    job = Job(id_=JobId(1), data=Input(1, 2, 3))

    # Check that there is no output
    assert hardware.get_job_output(job.id) is None

    try:
        # Submit the job
        hardware.submit_job(job)

        # Confirm that job status of job is RUNNING.
        assert hardware.get_job_status(job.id) == JobStatus.RUNNING

        # Confirm that an output is not ready yet.
        assert hardware.get_job_output(job.id) is None

        # Wait for the job to complete
        time.sleep(2)

        # Check job has completed
        assert hardware.get_job_status(job.id) == JobStatus.COMPLETED

        # Check expected output value
        assert hardware.get_job_output(job.id) == float(sum(job.data))
    finally:
        # Clean up remote job directory
        hardware.delete_remote_job_dir(job.id)
        pass


if __name__ == "__main__":
    args = get_command_line_args()
    ssh_config = read_json_config(args["ssh_config_path"])
    remote_script_config = read_json_config(args["remote_script_config_path"])
    run(ssh_config, remote_script_config)
