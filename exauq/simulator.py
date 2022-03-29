from abc import ABC, abstractmethod
from enum import Enum


class SimStatus(Enum):
    """
    Simulation status
    """
    STARTED = 0
    RUNNING = 1
    SUCCESS = 2
    RUN_FAILED = 3
    SUBMIT_FAILED = 4


class Simulator(ABC):
    """
     Class describing a single simulation with methods to run it, and 
     retrieve output data
    """
    def __init__(self):
        self.parameters = {}
        self.output_data = {}
        self.metadata = {}
        self.log_data = 'NULL'
        self.sup_data = 'NULL'

    @abstractmethod
    def run(self) -> None:
        """
        Method to run the simulator
        """
        pass

    @abstractmethod
    def sim_status(self) -> SimStatus:
        """
        Method to check current status of simulation
        """
        pass

    @abstractmethod
    def write_to_database(self) -> None:
        """
        Method to write simulation data to database
        """
        pass


class SimulatorFactory:
    """
    Class for generating simulators based on requested level/sim name
    """
    def __init__(self, available_simulators: dict):
        """

        :param available_simulators: dictionary of available simulators
        i.e. {"lvl0": PotatoSim, "lvl1": LaptopSim, "lvl2": HPCSim}
        """

        self.available_simulators = available_simulators

    def construct(self, simulator_id: str):
        """
        Constructs Simulator from Simulator Class identifier
        :param simulator_id: key/id of Simulator Class
        :return: Simulator
        """
        simulator = self.available_simulators[simulator_id]()
        return simulator
