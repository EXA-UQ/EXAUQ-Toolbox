import os
import pathlib
import time

from exauq.core.modelling import Input
from exauq.core.simulators import Simulator, SimulatorDomain
from tests.utilities.local_simulator import WORKSPACE, LocalSimulatorInterface

log_file = pathlib.Path("./simulations.csv")
try:
    # Initialise our interface to the local simulator, using the default workspace
    # directory.
    interface = LocalSimulatorInterface(WORKSPACE)

    # Initialise simulator with previously-created simulations log file
    domain = SimulatorDomain([(0, 1)] * 4)
    simulator = Simulator(domain, interface, log_file)

    # Evaluate some simulator inputs. Notice that these evaluate immediately, but each
    # returned output is None because the computations haven't finished.
    x1 = Input(0.1, 0.1, 0.1, 0.1)
    x2 = Input(0.2, 0.2, 0.2, 0.2)
    x3 = Input(0.3, 0.3, 0.3, 0.3)
    out1 = simulator.compute(x1)
    out2 = simulator.compute(x2)
    out3 = simulator.compute(x3)

    assert out1 is None
    assert out2 is None
    assert out3 is None

    # Wait for the simulations to complete
    # Note: this needs to account for (1) time to run the simulations (incl. any
    # delays in picking up the jobs to run) and (2) the polling period in the job
    # manager.
    time.sleep(15)

    # Check that the previous simulations now have output values
    assert all(sim[1] is not None for sim in simulator.previous_simulations)

finally:
    # Clean up simulations log file
    if log_file.exists():
        os.remove(log_file)
