import threading
import random
import queue
from exauq.simulator import SimStatus, Simulator, SimulatorFactory

class Scheduler:
    """
    Schedules and tracks simulation runs
    """
    def __init__(self, simulator_factory: SimulatorFactory):
        
        self.simulator_factory = simulator_factory
        self.submitted_job_list = []
        self.scheduler_thread = threading.Thread(target=self.run_scheduler)
        self._lock = threading.Lock()
        self.shutdown_scheduler = False

        self.requested_job_queue = queue.Queue()
        self.returned_job_queue = queue.Queue()

    def start_up(self):
        print("Start up the scheduler ... ")
        self.scheduler_thread.start()

    def shutdown(self):
        print("Shutdown of the scheduler started ... ")
        with self._lock:
            self.shutdown_scheduler = True
        self.scheduler_thread.join()
        print("Shutdown of the scheduler completed ... ")

    def run_scheduler(self) -> None:
        """
        Starts up scheduler main loop which checks if anything
        is in the requested jobs queue ready to be submitted. Any completed jobs
        are added to the returned_job_queue. The loop ends if the shutdown 
        signal is set and the requested job queue is empty.
        """
        while True:
            with self._lock:
                if not self.requested_job_queue.empty():
                    self.submit_job(self.requested_job_queue.get())
            for sim in self.submitted_job_list:
                if sim.sim_status() == SimStatus.SUCCESS or \
                   sim.sim_status() == SimStatus.RUN_FAILED:
                   print("Sim id {} of sim type {} has completed".format(
                       sim.metadata['simulation_id'], 
                       sim.metadata['simulation_type']))
                   sim.write_to_database()
                   self.returned_job_queue.put(sim)
            with self._lock:
                if self.shutdown_scheduler and self.requested_job_queue.empty():
                    break

    def request_job(self, parameters: dict, sim_type: str) -> int:
        """
        Request a new job given a set of input parameters and the
        simulation type        
        """
        sim_id = random.randint(1,1000)
        sim = self.simulator_factory.construct(sim_type)
        sim.parameters = parameters
        sim.metadata['simulation_id'] = sim_id
        sim.metadata['simulation_type'] = sim_type
        print("Adding simulation id {} of sim type {} to requested job queue".
            format(sim_id,sim_type))
        with self._lock:
            self.requested_job_queue.put(sim)
        return sim_id

    def submit_job(self, sim: Simulator) -> None:
        """
        Submits a simulation job
        """
        print("Submitting job for sim id {} of sim type {}".format(
            sim.metadata['simulation_id'], sim.metadata['simulation_type']))
        sim.run()
        self.submitted_job_list.append(sim)