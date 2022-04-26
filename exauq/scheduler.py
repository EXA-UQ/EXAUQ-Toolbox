import threading
import random
import queue
import time
from exauq.simulator import SimStatus, Simulator, SimulatorFactory

class Scheduler:
    """
    Schedules and tracks simulation runs
    """
    def __init__(self, simulator_factory: SimulatorFactory):
        
        self.simulator_factory = simulator_factory

        self.requested_job_queue = queue.Queue()
        self.submitted_job_list = []
        self.returned_job_queue = queue.Queue()

        self.scheduler_thread = threading.Thread(target=self.run_scheduler)
        self.monitor_thread = threading.Thread(target=self.monitor_status)
        self._lock = threading.Lock()

        self.shutdown_scheduler = False
        self.shutdown_monitoring = False

    def start_up(self):
        print("Start up the scheduler ... ")
        self.scheduler_thread.start()
        self.monitor_thread.start()

    def shutdown(self):
        print("Shutdown of the scheduler started ... ")
        with self._lock:
            self.shutdown_scheduler = True
        self.scheduler_thread.join()
        self.monitor_thread.join()
        print("Shutdown of the scheduler completed ... ")

    def run_scheduler(self, sleep_period = 10) -> None:
        """
        Starts up scheduler main loop which checks if anything
        is in the requested jobs queue ready to be submitted. Any completed jobs
        are added to the returned_job_queue. The loop ends if the shutdown 
        signal is set and the requested job queue is empty.
        """
        while True:
            # Submit all jobs in the current requested job queue
            with self._lock:
                while not self.requested_job_queue.empty():
                    self.submit_job(self.requested_job_queue.get())
                    time.sleep(2.0) # Fixed submit delay.
 
            # Check which of submitted jobs have completed and add them to
            # the returned_job_queue. For all successfully completed jobs,
            # write data to database
            with self._lock:
                for sim in self.submitted_job_list:
                    if sim.status == SimStatus.SUCCESS or \
                       sim.status == SimStatus.RUN_FAILED:
                       print("Sim id {} of sim type {} has completed".format(
                        sim.metadata['simulation_id'], 
                        sim.metadata['simulation_type']))
                       self.returned_job_queue.put(sim)
                    if sim.status == SimStatus.SUCCESS:
                        sim.write_to_database()

            # Shutdown scheduler main loop if shutdown signal has been recieved
            # and all current submitted jobs have been completed
            with self._lock:
                if self.shutdown_scheduler and self.requested_job_queue.empty():
                    all_runs_completed = all(sim.status == SimStatus.SUCCESS or
                    sim.status == SimStatus.RUN_FAILED for sim in self.submitted_job_list)
                    if all_runs_completed:
                        self.shutdown_monitoring = True
                        break

            # Delay scheduler between checks       
            time.sleep(sleep_period)

    def monitor_status(self, polling_period=10) -> None:
        """
        This routine checks and sets status of submitted simulator jobs.
        """
        while True:
            with self._lock:
                for sim in self.submitted_job_list:
                    # Poll status of submitted jobs if the current status of the 
                    # jobs indicates they have not completed
                    if sim.status != SimStatus.SUCCESS or \
                       sim.status != SimStatus.RUN_FAILED:
                        sim.sim_status()
            with self._lock:
                if self.shutdown_monitoring:
                    break
 
            # Delay till next status polling request
            time.sleep(polling_period)


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