import threading
import random
import queue
import time
from exauq.core.simulator import SimulatorFactory
from exauq.core.event import Event, EventType
from exauq.utilities.jobstatus import JobStatus


class Scheduler:
    """
    Schedules and tracks simulation runs. Two threads are spun up:

    - the main scheduler thread that periodically process the event queue
    for any new or yet unprocessed events
    - the monitoring thread that periodically pushes poll requests to the
    event queue

    These two processes are run in seperate threads as they don't necessarily
    have the same periods.
    """

    def __init__(
        self, simulator_factory: SimulatorFactory, scheduler_period=5, polling_period=10
    ):

        self.simulator_factory = simulator_factory

        self.event_queue = queue.Queue()
        self.requested_job_list = []
        self.requested_job_status = {}

        self.scheduler_period = scheduler_period
        self.polling_period = polling_period

        self.scheduler_thread = threading.Thread(target=self.run_scheduler)
        self.monitor_thread = threading.Thread(target=self.run_monitor)
        self._lock = threading.Lock()

        self.shutdown_signal = False
        self.shutdown_monitoring = False

        self.log_file = None

    def start_up(self):
        self.log_file = open("scheduler.log", "w", buffering=1)
        self.scheduler_thread.start()
        self.monitor_thread.start()

    def shutdown(self):
        with self._lock:
            self.shutdown_signal = True
        self.scheduler_thread.join()
        self.monitor_thread.join()
        self.log_file.close()

    def run_scheduler(self) -> None:
        """
        Starts up scheduler main loop which simply process all requested events
        and updates the status log. Shutdown scheduler main loop if shutdown
        signal has been received and all current requested jobs have completed.
        """
        while True:
            with self._lock:
                self.event_handler()
                self.update_status_log()
                shutdown_scheduler = self.shutdown_signal and all(
                    sim.JOBHANDLER.job_status == JobStatus.SUCCESS
                    or sim.JOBHANDLER.job_status == JobStatus.FAILED
                    for sim in self.requested_job_list
                )
                if shutdown_scheduler:
                    self.shutdown_monitoring = True
                    break
            time.sleep(self.scheduler_period)

    def run_monitor(self) -> None:
        """
        This routine periodically push poll requests to event list. Shutdown
        monitoring if the monitoring shutdown handle is True
        """
        while True:
            with self._lock:
                for sim in self.requested_job_list:
                    status = sim.JOBHANDLER.job_status
                    # Only push poll request for sim jobs that hasn't failed in some way
                    # or hasn't successfully completed already.
                    poll = (
                        status != JobStatus.INSTALL_FAILED
                        and status != JobStatus.SUBMIT_FAILED
                        and status != JobStatus.SUCCESS
                        and status != JobStatus.FAILED
                    )
                    if poll:
                        self.event_queue.put(Event(EventType.POLL_SIM, sim))
                if self.shutdown_monitoring:
                    break
            time.sleep(self.polling_period)

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
            self.event_queue.put(Event(EventType.INSTALL_SIM, sim))
            self.event_queue.put(Event(EventType.RUN_SIM, sim))

    def event_handler(self) -> None:
        """
        Handle events in the event queue.
        """
        # Loop over all events and process those that is ready to be processed
        passed_over_events = []
        while not self.event_queue.empty():
            event = self.event_queue.get()
            sim = event.sim
            sim_id = sim.metadata["simulation_id"]
            if event.type == EventType.INSTALL_SIM:
                sim.JOBHANDLER.install_sim(sim_id)
            elif event.type == EventType.RUN_SIM:
                # Execute a run event if the sim is already installed,
                # otherwise skip over the run event
                if sim.JOBHANDLER.job_status == JobStatus.INSTALLED:
                    sim.JOBHANDLER.run_sim(sim.COMMAND)
                else:
                    passed_over_events.append(event)
            elif event.type == EventType.POLL_SIM:
                sim.JOBHANDLER.poll_sim()

        # Add passed over events back on the event queue
        for event in passed_over_events:
            self.event_queue.put(event)

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
            self.requested_job_status[sim_id]["job_status"] = sim.JOBHANDLER.job_status
            self.requested_job_status[sim_id][
                "submit_time"
            ] = sim.JOBHANDLER.submit_time
            self.requested_job_status[sim_id][
                "last_poll_time"
            ] = sim.JOBHANDLER.last_poll_time
        self.log_status()

    def log_status(self) -> None:
        """
        Simple event printout of current status
        """
        current_time = time.strftime("%H:%M:%S", time.localtime())
        message = current_time + ":\n"
        for sim_id, sim_status in self.requested_job_status.items():
            message = (
                message
                + "sim_id: {0} host: {1} job_id: {2} submit_time: {3} last_poll_time: {4} job_status: {5}\n".format(
                    sim_id,
                    sim_status["host"],
                    sim_status["job_id"],
                    sim_status["submit_time"],
                    sim_status["last_poll_time"],
                    sim_status["job_status"],
                )
            )
        print(message)
        self.log_file.write(message)
