from exauq.core.modelling import Experiment
from exauq.core.designers import SingleLevelAdaptiveSampler
from tests.unit.fakes import DumbEmulator, OneDimSimulator


# The following simulator represents the function f:[0, 1] -> R, f(x) = x
simulator = OneDimSimulator(0, 1)

# Initialise an emulator (the following is a fake implementation of an emulator)
emulator = DumbEmulator()

# Create an initial design of points from the interval [0, 1]
initial_design = [Experiment(0.2),
                  Experiment(0.4),
                  Experiment(0.6),
                  Experiment(0.8)]

# Initialise a 'designer' object that encapsulates the adaptive sampling
# methodology
designer = SingleLevelAdaptiveSampler(initial_design=initial_design)

# Create a new emulator trained with simulator outputs with the designer
trained_emulator = designer.train(emulator, simulator)

# View the data the emulator is trained on
print("Training data:", trained_emulator.training_data, '\n')

# Make some predictions from the trained emulator:
print("Prediction at x = 0.1:", trained_emulator.predict(Experiment(0.1)))
print("Prediction at x = 0.2:", trained_emulator.predict(Experiment(0.2)))
