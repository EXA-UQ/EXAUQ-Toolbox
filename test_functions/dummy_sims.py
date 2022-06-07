import time
from exauq.core.simulator import Simulator
from exauq.utilities.JobStatus import JobStatus
from exauq.utilities.BgHandler import BgHandler

class DummySimLvl0(Simulator):
    """
    Simple level 0 dummy simulator
    """
    JOBHANDLER = BgHandler(host="localhost", user="")
    COMMAND = "sleep 1"

    def run(self) -> None:
        self.metadata['job_id'] = self.JOBHANDLER.submit_job(sim_id=self.metadata['simulation_id'], command=self.COMMAND)

    def sim_status(self) -> JobStatus:
        status = self.JOBHANDLER.poll_job(sim_id=self.metadata['simulation_id'], job_id=self.metadata['job_id'])
        if status:
            self.status = status 
        return self.status

    def write_to_database(self) -> None:
        print("Writing sim id {} of sim type {} data to database".format(
            self.metadata['simulation_id'], self.metadata['simulation_type']))

class DummySimLvl1(Simulator):
    """
    Simple level 1 dummy simulator
    """
    JOBHANDLER = BgHandler(host="localhost", user="")
    COMMAND = "sleep 2"

    def run(self) -> None:
        self.metadata['job_id'] = self.JOBHANDLER.submit_job(sim_id=self.metadata['simulation_id'], command=self.COMMAND)

    def sim_status(self) -> JobStatus:
        status = self.JOBHANDLER.poll_job(sim_id=self.metadata['simulation_id'], job_id=self.metadata['job_id'])
        if status:
            self.status = status 
        return self.status

    def write_to_database(self) -> None:
        print("Writing sim id {} of sim type {} data to database".format(
            self.metadata['simulation_id'], self.metadata['simulation_type']))

class DummySimLvl2(Simulator):
    """
    Simple level 2 dummy simulator
    """
    JOBHANDLER = BgHandler(host="localhost", user="")
    COMMAND = "sleep 5"

    def run(self) -> None:
        self.metadata['job_id'] = self.JOBHANDLER.submit_job(sim_id=self.metadata['simulation_id'], command=self.COMMAND)

    def sim_status(self) -> JobStatus:
        status = self.JOBHANDLER.poll_job(sim_id=self.metadata['simulation_id'], job_id=self.metadata['job_id'])
        if status:
            self.status = status 
        return self.status

    def write_to_database(self) -> None:
        print("Writing sim id {} of sim type {} data to database".format(
            self.metadata['simulation_id'], self.metadata['simulation_type']))

class DummySimLvl3(Simulator):
    """
    Simple level 3 dummy simulator
    """
    JOBHANDLER = BgHandler(host="localhost", user="")
    COMMAND = "sleep 10"

    def run(self) -> None:
        self.metadata['job_id'] = self.JOBHANDLER.submit_job(sim_id=self.metadata['simulation_id'], command=self.COMMAND)

    def sim_status(self) -> JobStatus:
        status = self.JOBHANDLER.poll_job(sim_id=self.metadata['simulation_id'], job_id=self.metadata['job_id'])
        if status:
            self.status = status 
        return self.status

    def write_to_database(self) -> None:
        print("Writing sim id {} of sim type {} data to database".format(
            self.metadata['simulation_id'], self.metadata['simulation_type']))
    