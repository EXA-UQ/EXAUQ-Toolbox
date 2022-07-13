import threading
import random
import queue
import time
from exauq.core.simulator import SimulatorFactory
from exauq.core.event import Event, EventType
from exauq.utilities.JobStatus import JobStatus


class Scheduler:
    """
    Schedules and tracks simulation runs
    """

    def __init__(self, simulator_factory: SimulatorFactory):

        self.simulator_factory = simulator_factory

        self.event_queue = queue.Queue()
        self.requested_job_list = []

        self.scheduler_thread = threading.Thread(target=self.run_scheduler)
        self.monitor_thread = threading.Thread(target=self.run_monitor)
        self._lock = threading.Lock()

        self.shutdown_scheduler = False
        self.shutdown_monitoring = False

        self.log_file = None

    def start_up(self):
        self.log_file = open("scheduler.log", "w", buffering=1)
        self.log_event("Start up the scheduler ... ")
        self.scheduler_thread.start()
        self.monitor_thread.start()

    def shutdown(self):
        self.log_event("Shutdown of the scheduler started ... ")
        with self._lock:
            self.shutdown_scheduler = True
        self.scheduler_thread.join()
        self.monitor_thread.join()
        self.log_event("Shutdown of the scheduler completed ... ")
        self.log_file.close()

    def run_scheduler(self, sleep_period=5) -> None:
        """
        Starts up scheduler main loop which simply process all requested events
        and updates the status log. Shutdown scheduler main loop if shutdown 
        signal has been recieved and all current requested jobs have completed.
        """
        while True:
            with self._lock:
                self.event_handler()
                self.update_status_log()
                if self.shutdown_scheduler and self.all_runs_completed():
                    self.shutdown_monitoring = True
                    break
            time.sleep(sleep_period)

    def run_monitor(self, polling_period=10) -> None:
        """
        This routine periodically push poll requests to event list. Shutdown
        monitoring main loop if shutdown signal has been recieved and all current 
        submitted jobs have been completed
        """
        while True:
            with self._lock:
                for sim in self.requested_job_list:
                    if (
                        sim.JOBHANDLER.job_status != JobStatus.SUCCESS
                        or sim.JOBHANDLER.job_status != JobStatus.FAILED
                    ):
                        self.event_queue.put(Event(EventType.POLL_SIM, sim))
                if self.shutdown_monitoring and self.all_runs_completed():
                    break
            time.sleep(polling_period)

    def request_job(self, parameters: dict, sim_type: str) -> None:
        """
        Request a new job given a set of input parameters and the
        simulation type and push request to event list
        """
        with self._lock:
            sim = self.simulator_factory.construct(sim_type)
            sim.parameters = parameters
            sim.metadata["simulation_id"] = str(random.randint(1, 1000))
            sim.metadata["simulation_type"] = sim_type
            self.requested_job_list.append(sim)
            self.event_queue.put(Event(EventType.SUBMIT_SIM, sim))
            self.log_event(
                "Added sim {} of type {} to event list for submission".format(
                    sim.metadata["simulation_id"], sim.metadata["simulation_type"]
                )
            )

    def event_handler(self) -> None:
        """
        Process all current events
        """
        while not self.event_queue.empty():
            event = self.event_queue.get()
            sim = event.sim
            if event.type == EventType.SUBMIT_SIM:
                sim.JOBHANDLER.submit_job(
                sim_id=sim.metadata["simulation_id"], command=sim.COMMAND
                )
                self.log_event(
                "Submitted job for sim {}".format(sim.metadata["simulation_id"])
                )
            elif event.type == EventType.POLL_SIM:
                sim.JOBHANDLER.poll_job(sim_id=sim.metadata["simulation_id"])
                self.log_event(
                    "Polling sim {0} with job id {1} for job_status.".format(
                        sim.metadata["simulation_id"],
                        sim.JOBHANDLER.job_id
                    )
                )

    def update_status_log(self):
        """
        Update the status log of all current requested sim runs
        """
        for sim in self.requested_job_list:
            self.log_event(
                "Sim {0} with job id {1} has job status {2}".format(
                    sim.metadata["simulation_id"],
                    sim.JOBHANDLER.job_id,
                    sim.JOBHANDLER.job_status  
                )
            )

    def all_runs_completed(self) -> bool:
        """
        Check if all runs in the submitted job list has completed
        """
        return all(
            sim.JOBHANDLER.job_status == JobStatus.SUCCESS
            or sim.JOBHANDLER.job_status == JobStatus.FAILED
            for sim in self.requested_job_list
        )

    def log_event(self, message: str) -> None:
        """
        Simple event printout to stdout with time_stamp
        """
        print(time.strftime("%H:%M:%S", time.localtime()) + " : " + message + "\n")
        self.log_file.write(
            time.strftime("%H:%M:%S", time.localtime()) + " : " + message + "\n"
        )
