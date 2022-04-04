import time
from exauq.simulator import SimStatus, Simulator

class DummySimLvl0(Simulator):
    """
    Simple level 0 dummy simulator
    """
    def run(self) -> None:
        time.sleep(1)
        self.status = SimStatus.SUCCESS

    def sim_status(self) -> SimStatus:
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
        self.status = SimStatus.SUCCESS

    def sim_status(self) -> SimStatus:
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
        self.status = SimStatus.SUCCESS

    def sim_status(self) -> SimStatus:
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
        self.status = SimStatus.SUCCESS

    def sim_status(self) -> SimStatus:
        return self.status

    def write_to_database(self) -> None:
        print("Writing sim id {} of sim type {} data to database".format(
            self.metadata['simulation_id'], self.metadata['simulation_type']))
    