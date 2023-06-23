"""An example script demonstrating how to work with wrappers of mogp-emulator objects.
"""

from exauq.core.modelling import TrainingDatum
from exauq.core.emulators import MogpEmulator
import numpy as np

# Initialise a wrapped mogp emulator, specifying a particular kernel function:
emulator = MogpEmulator(kernel="Matern52")

# Get the underlying mogp emulator. Note how it is initialised with no training
# data
gp = emulator.gp
print()
print(f"gp is of type {type(gp)}")
print(f"gp.inputs: {gp.inputs}")
print(f"gp.targets: {gp.targets}")

# Can create initial design and corresponding outputs from numpy arrays:
inputs = np.random.rand(10, 2)
outputs = 10 * np.random.rand(10)
data = TrainingDatum.list_from_arrays(inputs, outputs)
print("Training data:")
for datum in data:
    print("  ", datum)

# Fit emulator to the training data (with hyperparameter estimation):
emulator.fit(data)

# Look at fitted hyperparameters of underlying mogp emulator
print()
print("Hyperparameters estimated:\n", emulator.gp.theta, "\n")

# Now train on same data but with bounds on hyperparameters
bounds = (
    (0.5, np.inf),  # first corr length param
    (0.5, np.inf),  # second corr length param
    (None, None),  # covariance (unrestricted)
)
emulator.fit(data, hyperparameter_bounds=bounds)

# Look at estimated hyperparameters again
print()
print("Hyperparameters estimated (under bounds):\n", emulator.gp.theta, "\n")
