import random
import time
from exauq.core.scheduler import Scheduler
from exauq.core.simulator import SimulatorFactory
from exauq.utilities.JobStatus import JobStatus
from test_functions.dummy_sims import (
    DummySimLvl0,
    DummySimLvl1,
    DummySimLvl2,
    DummySimLvl3,
)



job_status = {"123456789": { "host": "localhost",
                             "job_id": "Hgs562353vd",
                             "job_status": JobStatus.SUCCESS,
                             "submit_time": "12:35:01",
                             "last_poll_time": "13:02:54"}
              }


if __name__ == "__main__":
    """
    This tests the scheduler by running the four dummy simulations. The test
    passes if all the submitted jobs has completed successfully. Send shutdown
    signal to the scheduler once the requested job queue is cleared.
    """
    sim_types = ["lvl0", "lvl1", "lvl2", "lvl3"]
    sim_factory = SimulatorFactory(
        {
            "lvl0": DummySimLvl0,
            "lvl1": DummySimLvl1,
            "lvl2": DummySimLvl2,
            "lvl3": DummySimLvl3,
        }
    )
    scheduler = Scheduler(simulator_factory=sim_factory)
    scheduler.start_up()

    for i in range(20):
        sim_type = random.choice(sim_types)
        scheduler.request_job(parameters={}, sim_type=sim_type)

    scheduler.shutdown()
