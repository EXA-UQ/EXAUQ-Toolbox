from random import random
from simulator import SimStatus, Simulator, SimulatorFactory


class Scheduler:
    """
    Schedules and tracks simulation runs
    """
    def __init__(self, simulator_factory: SimulatorFactory):
        
        self.simulator_factory = simulator_factory
        self.requested_jobs = []
        self.submitted_jobs = []
        self.returned_jobs = []
        self.failed_jobs = []

    def start_up(self) -> None:
        """
        Starts up scheduler main loop which checks if anything
        is in the requested jobs list ready to be submitted and
        remove any returned jobs form the submitted list. 
        """
        while True:
            if self.requested_jobs:
                self.submit_job(self.requested_jobs.pop(0))
            for sim in self.submitted_jobs:
                if sim.sim_status() == SimStatus.SUCCESS:
                   sim.write_to_database()
                   self.returned_jobs.append(sim)
                if sim.sim_status() == SimStatus.RUN_FAILED:
                   self.failed_jobs.append(sim)
                   self.returned_jobs.append(sim)
                   
            for sim in self.returned_jobs:
                self.submitted_jobs.remove(sim)

    def request_job(self, parameters: dict, sim_type: str) -> int:
        """
        Request a new job given a set of input parameters and the
        simulation type        
        """
        sim_id = int(1000*random())
        sim = self.simulator_factory.construct(sim_type)
        sim.parameters = parameters
        sim.metadata['simulation_id'] = sim_id
        sim.metadata['simulation_type'] = sim_type
        self.requested_jobs.append(sim)
        return sim_id

    def submit_job(self, sim: Simulator) -> None:
        """
        Submits a simulation job
        """
        sim.run()
        self.submitted_jobs.append(sim)