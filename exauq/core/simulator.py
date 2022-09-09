from abc import ABC, abstractmethod
from typing import TypeVar

TSimulator = TypeVar("TSimulator", bound="Simulator")


class Simulator(ABC):
    """
    Class describing a single simulation with methods to run it, and
    retrieve output data
    """

    def __init__(self):
        self.parameters = {}
        self.output_data = {}
        self.metadata = {}
        self.log_data = "NULL"
        self.sup_data = "NULL"
        self.JOBHANDLER = None
        self.COMMAND = None


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

    def construct(self, simulator_id: str) -> TSimulator:
        """
        Constructs Simulator from Simulator Class identifier
        :param simulator_id: key/id of Simulator Class
        :return: Simulator
        """
        simulator = self.available_simulators[simulator_id]()
        return simulator
