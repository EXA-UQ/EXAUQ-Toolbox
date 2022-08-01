import threading
import random
import queue
import time

from multiprocessing import Process, Pipe

from exauq.core.simulator import SimulatorFactory
from exauq.core.event import Event, EventType
from exauq.gui.gui_app import start_dash
from exauq.utilities.JobStatus import JobStatus


class Scheduler:
    """
    Schedules and tracks simulation runs
    """

    def __init__(self, simulator_factory: SimulatorFactory):

        self.simulator_factory = simulator_factory

        self.event_queue = queue.Queue()
        self.requested_job_list = []
        self.requested_job_status = {}

        self.scheduler_thread = threading.Thread(target=self.run_scheduler)
        self.monitor_thread = threading.Thread(target=self.run_monitor)
        self._lock = threading.Lock()

        self.shutdown_scheduler = False
        self.shutdown_monitoring = False

        self.frontend = False
        self.frontend_conn = None
        self.frontend_process = None

        self.log_file = None

    def start_up(self, frontend=True):
        self.frontend = frontend
        if self.frontend:
            self.run_frontend()

        self.log_file = open("scheduler.log", "w", buffering=1)
        self.scheduler_thread.start()
        self.monitor_thread.start()

    def shutdown(self):
        with self._lock:
            self.shutdown_scheduler = True
        self.scheduler_thread.join()
        self.monitor_thread.join()
        self.log_file.close()

    def run_frontend(self) -> None:
        """
        Runs frontend...
        """
        self.frontend_conn, child_conn = Pipe()
        self.frontend_process = Process(target=start_dash, args=(child_conn,))
        self.frontend_process.start()

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
                    status = sim.JOBHANDLER.job_status
                    if status != JobStatus.SUCCESS or status != JobStatus.FAILED:
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

    def event_handler(self) -> None:
        """
        Process all current events
        """
        while not self.event_queue.empty():
            event = self.event_queue.get()
            sim = event.sim
            if event.type == EventType.SUBMIT_SIM:
                sim.JOBHANDLER.submit_job(sim_id=sim.metadata["simulation_id"], command=sim.COMMAND)
            elif event.type == EventType.POLL_SIM:
                sim.JOBHANDLER.poll_job(sim_id=sim.metadata["simulation_id"])

    def update_status_log(self):
        """
        Update the status log of all current requested sim runs
        """
        for sim in self.requested_job_list:
            sim_id = sim.metadata["simulation_id"]
            if sim_id not in self.requested_job_status:
                self.requested_job_status[sim_id] = {}
            self.requested_job_status[sim_id]["host"] = sim.JOBHANDLER.host
            self.requested_job_status[sim_id]["job_id"] = sim.JOBHANDLER.job_id
            self.requested_job_status[sim_id]["job_status"] =  sim.JOBHANDLER.job_status
            self.requested_job_status[sim_id]["submit_time"] =  sim.JOBHANDLER.submit_time
            self.requested_job_status[sim_id]["last_poll_time"] =  sim.JOBHANDLER.last_poll_time
        self.log_status()

    def all_runs_completed(self) -> bool:
        """
        Check if all runs in the submitted job list has completed
        """
        return all(
            sim.JOBHANDLER.job_status == JobStatus.SUCCESS
            or sim.JOBHANDLER.job_status == JobStatus.FAILED
            for sim in self.requested_job_list
        )

    def log_status(self) -> None:
        """
        Simple event printout of current status
        """
        current_time = time.strftime("%H:%M:%S", time.localtime())
        message = current_time + ":\n"
        for sim_id, sim_status in self.requested_job_status.items():
            message = message + \
                "sim_id: {0} host: {1} job_id: {2} submit_time: {3} last_poll_time: {4} job_status: {5}\n".format(
                    sim_id,
                    sim_status["host"],
                    sim_status["job_id"],
                    sim_status["submit_time"],
                    sim_status["last_poll_time"],
                    sim_status["job_status"]
                )
        print(message)
        self.log_file.write(message)
