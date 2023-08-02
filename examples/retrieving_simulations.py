import math
import time
import timeit

from exauq.core.modelling import Input
from exauq.core.simulators import Simulator
from tests.unit.fakes import InMemoryHardware


# Initialise a pretend piece of hardware, which computes a 'costly' function in memory.
# For real use cases, a class implementing the HardwareInterface abstract class would
# instead be used.
def f(x: Input):
    time.sleep(1)
    return x[1] + (x[0] ** 2) + (x[1] ** 2) - math.sqrt(2)

hardware = InMemoryHardware(f)

# Initialise simulator with previously-created simulations log file
simulator = Simulator(hardware, "./simulations.csv")

# Evaluate some simulator inputs. Notice that these evaluate immediately, but each
# returned output is None because the computations haven't finished.
x1 = Input(0.1, 0.1)
x2 = Input(0.3, 0.3)
x3 = Input(0.6, 0.8)
out1 = simulator.compute(x1)
out2 = simulator.compute(x2)
out3 = simulator.compute(x3)

assert out1 is None
assert out2 is None
assert out3 is None

# Wait for the simulations to complete
time.sleep(5)

# View the previous simulations
print("Previous simulations:")
for simulation in simulator.previous_simulations:
    print(simulation)

# The simulator object evaluates previously-computed inputs quickly, because the
# simulation is cached:
print(f"Simulator value at {x1}: {simulator.compute(x1)}")
timeit.timeit("simulator.compute(x1)", globals=globals(), number=1)
