import threading
import random
import queue
import time
from exauq.core.simulator import Simulator, SimulatorFactory
from exauq.utilities.JobStatus import JobStatus

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


    def run_scheduler(self, sleep_period = 5) -> None:
        """
        Starts up scheduler main loop which checks if anything
        is in the requested jobs queue ready to be submitted. Any completed jobs
        are added to the returned_job_queue. Shutdown scheduler main loop if 
        shutdown signal has been recieved and all current submitted jobs have 
        been completed.
        """
        while True:
            with self._lock:
                self.submit_jobs()
 
                for sim in self.submitted_job_list:
                    if sim.JOBHANDLER.job_id is None:
                        sim.JOBHANDLER.get_jobid()
                    if sim.JOBHANDLER.job_status == JobStatus.SUCCESS:
                        print("Sim id {} has succeed".format(sim.metadata['simulation_id']))
                        self.returned_job_queue.put(sim)
                    if sim.JOBHANDLER.job_status == JobStatus.FAILED:
                        print("Sim id {} has failed".format(sim.metadata['simulation_id']))

                if self.shutdown_scheduler and self.all_runs_completed():
                    self.shutdown_monitoring = True
                    break
      
            time.sleep(sleep_period)


    def monitor_status(self, polling_period=10) -> None:
        """
        This routine periodically polls the status of submitted simulator jobs. Shutdown 
        monitoring main loop if shutdown signal has been recieved and all current submitted 
        jobs have been completed
        """
        while True:
            with self._lock:

                for sim in self.submitted_job_list:
                    if sim.JOBHANDLER.job_status != JobStatus.SUCCESS or sim.JOBHANDLER.job_status != JobStatus.FAILED:
                        sim.JOBHANDLER.poll_job(sim_id=sim.metadata['simulation_id'])

                if self.shutdown_monitoring and self.all_runs_completed():
                    break
                
            time.sleep(polling_period)


    def request_job(self, parameters: dict, sim_type: str) -> None:
        """
        Request a new job given a set of input parameters and the
        simulation type        
        """
        with self._lock:
            sim = self.simulator_factory.construct(sim_type)
            sim.parameters = parameters
            sim.metadata['simulation_id'] = str(random.randint(1,1000))
            sim.metadata['simulation_type'] = sim_type
            self.requested_job_queue.put(sim)
            print("Added sim {} of type {} to requested job queue".format(
                sim.metadata['simulation_id'], sim.metadata['simulation_type']))


    def submit_jobs(self) -> None:
        """
        Submits current jobs in requested job queue
        """
        while not self.requested_job_queue.empty():
            sim = self.requested_job_queue.get()
            sim.JOBHANDLER.submit_job(sim_id=sim.metadata['simulation_id'], command=sim.COMMAND)
            self.submitted_job_list.append(sim)
            print("Submitted job for sim {}".format(sim.metadata['simulation_id']))


    def all_runs_completed(self) -> bool:
        """
        Check if all runs in the submitted job list has completed
        """
        return all(sim.JOBHANDLER.job_status == JobStatus.SUCCESS or sim.JOBHANDLER.job_status == JobStatus.FAILED for sim in self.submitted_job_list)