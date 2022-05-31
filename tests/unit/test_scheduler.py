import pytest
from exauq.core.scheduler import Scheduler
from exauq.core.simulator import SimulatorFactory
from exauq.utilities.JobStatus import JobStatus
from test_functions.dummy_sims import DummySimLvl0, DummySimLvl1, DummySimLvl2, DummySimLvl3

def test_scheduler() -> None:
    """
    This tests the scheduler by running the four dummy simulations. The test
    passes if all the submitted jobs has completed successfully. Send shutdown
    signal to the scheduler once the requested job queue is cleared.
    """
    sim_factory = SimulatorFactory({
                  "lvl0" : DummySimLvl0,
                  "lvl1" : DummySimLvl1,
                  "lvl2" : DummySimLvl2,
                  "lvl3" : DummySimLvl3           
    })
    scheduler=Scheduler(simulator_factory=sim_factory)
    scheduler.start_up()
    list_of_jobs = [
        ({}, "lvl0"),
        ({}, "lvl1"),
        ({}, "lvl2"),
        ({}, "lvl3")
    ]
    for job in list_of_jobs:
        scheduler.request_job(parameters=job[0], sim_type=job[1])
    while True:
        if scheduler.requested_job_queue.empty():
            scheduler.shutdown()
            break
    success = all(sim.status == JobStatus.SUCCESS for sim in scheduler.submitted_job_list)
    assert success is True