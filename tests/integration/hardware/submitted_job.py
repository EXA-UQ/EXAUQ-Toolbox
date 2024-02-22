import json
import pathlib
import sys
from typing import Any

from exauq.core.modelling import Input
from exauq.sim_management.hardware import JobStatus, RemoteServerScript
from exauq.sim_management.jobs import Job, JobId


def get_ssh_config_path() -> str:
    try:
        return sys.argv[1]
    except IndexError:
        print(
            f"{sys.argv[0]} error: No path to a ssh config file supplied.",
            file=sys.stderr,
        )
        sys.exit(1)


def read_ssh_config(path: str) -> dict[str, Any]:
    with open(pathlib.Path(path), mode="r") as ssh_config_file:
        return json.load(ssh_config_file)


if __name__ == "__main__":
    config_path = get_ssh_config_path()
    ssh_config = read_ssh_config(config_path)

    # Create interface to remote server where we can run a script
    hardware = RemoteServerScript(**ssh_config)

    # Create a job to submit
    job = Job(id_=JobId(1), data=Input(1, 2, 3))

    # First check that the job is PENDING before submitting
    assert hardware.get_job_status(job.id) == JobStatus.PENDING

    # Submit the job
    hardware.submit_job(job)

    # Confirm that job status of job is SUBMITTED
    assert hardware.get_job_status(job.id) == JobStatus.SUBMITTED
