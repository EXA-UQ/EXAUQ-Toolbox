from exauq.core.simulator import Simulator
from exauq.utilities.JobStatus import JobStatus
from math import sqrt, pi, sin

"""
A Simple 2D, 2 Level Toy Problem as defined in... 
"""

problem_parameters = {"x1": (0, 1),
                      "x2": (0, 1)}

problem_outputs = {"y": None}


def toy_2d_lvl0(x1: float, x2: float) -> float:
    # Check 0 < x1&x2 < 1
    if not 0 <= x1 <= 1 and 0 <= x2 <= 1:
        raise ValueError("ERROR: input variable(s) out of range")

    return x2 + x1**2 + x2**2 - sqrt(2)


def toy_2d_lvl1(x1: float, x2: float) -> float:
    if not 0 <= x1 <= 1 and 0 <= x2 <= 1:
        raise ValueError("ERROR: input variable(s) out of range")

    return toy_2d_lvl0(x1, x2) + sin(2*pi*x1) + sin(4*pi*x1*x2)


class Simple2DLvl0(Simulator):
    """
    Simple 2D Test Function
    """
    def run(self) -> None:
        """
        Runs toy_2d_lvl0 function
        """
        x1 = self.parameters['x1']
        x2 = self.parameters['x2']

        y = toy_2d_lvl0(x1, x2)

        self.output_data['y'] = y
        self.status = JobStatus.SUCCESS

    def sim_status(self) -> int:
        """
        Method to check current status of simulation
        """
        return self.status

    def write_to_database(self) -> None:
        """
        Method to write simulation data to database
        """
        pass


class Simple2DLvl1(Simulator):
    """
    Simple 2D Test Function
    """
    def run(self) -> None:
        """
        Runs toy_2d_lvl1 function
        """
        x1 = self.parameters['x1']
        x2 = self.parameters['x2']

        y = toy_2d_lvl1(x1, x2)

        self.output_data['y'] = y
        self.status = JobStatus.SUCCESS

    def sim_status(self) -> int:
        """
        Method to check current status of simulation
        """
        return self.status

    def write_to_database(self) -> None:
        """
        Method to write simulation data to database
        """
        pass