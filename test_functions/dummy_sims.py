import time
import subprocess
from exauq.core.simulator import Simulator
from exauq.utilities.JobStatus import JobStatus

class DummySimLvl0(Simulator):
    """
    Simple level 0 dummy simulator
    """
    
    def run(self) -> None:
        time.sleep(1)
        self.status = JobStatus.SUCCESS

    def sim_status(self) -> JobStatus:
        return self.status

    def write_to_database(self) -> None:
        print("Writing sim id {} of sim type {} data to database".format(
            self.metadata['simulation_id'], self.metadata['simulation_type']))

class DummySimLvl1(Simulator):
    """
    Simple level 1 dummy simulator
    """
    __test__ = False
    def run(self) -> None:
        time.sleep(2)
        self.status = JobStatus.SUCCESS

    def sim_status(self) -> JobStatus:
        return self.status

    def write_to_database(self) -> None:
        print("Writing sim id {} of sim type {} data to database".format(
            self.metadata['simulation_id'], self.metadata['simulation_type']))

class DummySimLvl2(Simulator):
    """
    Simple level 2 dummy simulator
    """
    def run(self) -> None:
        time.sleep(5)
        self.status = JobStatus.SUCCESS

    def sim_status(self) -> JobStatus:
        return self.status

    def write_to_database(self) -> None:
        print("Writing sim id {} of sim type {} data to database".format(
            self.metadata['simulation_id'], self.metadata['simulation_type']))

class DummySimLvl3(Simulator):
    """
    Simple level 3 dummy simulator
    """
    def run(self) -> None:
        time.sleep(10)
        self.status = JobStatus.SUCCESS

    def sim_status(self) -> JobStatus:
        return self.status

    def write_to_database(self) -> None:
        print("Writing sim id {} of sim type {} data to database".format(
            self.metadata['simulation_id'], self.metadata['simulation_type']))
    