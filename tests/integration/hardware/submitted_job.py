from exauq.core.modelling import Input
from exauq.sim_management.hardware import JobStatus, RemoteServerScript
from exauq.sim_management.jobs import Job, JobId
from tests.integration.hardware.utilities import get_ssh_config_path, read_ssh_config

if __name__ == "__main__":
    config_path = get_ssh_config_path()
    ssh_config = read_ssh_config(config_path)

    # Create interface to remote server where we can run a script
    hardware = RemoteServerScript(**ssh_config)

    # Create a job to submit
    job = Job(id_=JobId(1), data=[Input(1, 2, 3)])

    # First check that the job is NOT_SUBMITTED before submitting
    assert hardware.get_job_status(job.id) == JobStatus.NOT_SUBMITTED

    # Submit the job
    hardware.submit_job(job)

    # Confirm that job status of job is SUBMITTED
    assert hardware.get_job_status(job.id) == JobStatus.SUBMITTED
