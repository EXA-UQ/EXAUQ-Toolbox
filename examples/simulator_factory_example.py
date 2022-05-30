from exauq.core.simulator import SimulatorFactory, Simulator
from exauq.core.scheduler import Scheduler
from exauq.utilities.JobStatus import JobStatus


class PotatoSim(Simulator):
    """
    Simulator run on a potato
    """
    def run(self) -> None:
        """
        Method to run the simulator
        """
        print("Running the potato simulator")

    def sim_status(self) -> JobStatus:
        """
        Method to check current status of simulation
        """
        return JobStatus.RUNNING

    def write_to_database(self) -> None:
        """
        Method to write simulation data to database
        """
        print("Writing results to the database")


class LaptopSim(Simulator):
    """
    Simulator run on a Laptop
    """
    def run(self) -> None:
        """
        Method to run the simulator
        """
        print("Running the Laptop simulator")

    def sim_status(self) -> JobStatus:
        """
        Method to check current status of simulation
        """
        return JobStatus.RUNNING

    def write_to_database(self) -> None:
        """
        Method to write simulation data to database
        """
        print("Writing results to the database")


class HPCSim(Simulator):
    """
    Simulator run on a HPC
    """
    def run(self) -> None:
        """
        Method to run the simulator
        """
        print("Running the HPC simulator")

    def sim_status(self) -> JobStatus:
        """
        Method to check current status of simulation
        """
        return JobStatus.RUNNING

    def write_to_database(self) -> None:
        """
        Method to write simulation data to database
        """
        print("Writing results to the database")


if __name__ == "__main__":
    """Simulator Factory Usage"""
    sim_factory = SimulatorFactory({"lvl0": PotatoSim, "lvl1": LaptopSim, "lvl2": HPCSim})

    potato_simulator = sim_factory.construct("lvl0")
    laptop_simulator = sim_factory.construct("lvl1")
    hpc_simulator = sim_factory.construct("lvl2")

    potato_simulator.run()
    laptop_simulator.run()
    hpc_simulator.run()

    """Init Schedular"""
    schedular = Scheduler(sim_factory)
    # schedular.start_up()
    lvl0_job_id = schedular.request_job({"input_01": 2.5, "input_02": 1.25}, "lvl0")
    lvl1_job_id = schedular.request_job({"input_01": 5.125, "input_02": 3.21}, "lvl1")
    lvl2_job_id = schedular.request_job({"input_01": 4.32, "input_02": 0.25}, "lvl2")