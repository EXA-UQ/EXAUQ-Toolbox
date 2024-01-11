from exauq.core.designers import SingleLevelAdaptiveSampler
from exauq.core.modelling import Input, TrainingDatum
from tests.unit.fakes import FakeGP, OneDimSimulator

# The following simulator represents the function f:[0, 1] -> R, f(x) = x
simulator = OneDimSimulator(0, 1)

# Initialise an emulator (the following is a fake implementation of an emulator)
emulator = FakeGP()

# Create an initial design of points from the interval [0, 1]
initial_data = [
    TrainingDatum(Input(0.2), 0.2),
    TrainingDatum(Input(0.4), 0.4),
    TrainingDatum(Input(0.6), 0.6),
    TrainingDatum(Input(0.8), 0.8),
]

# Initialise a 'designer' object that encapsulates the adaptive sampling
# methodology
designer = SingleLevelAdaptiveSampler(initial_data=initial_data)

# Create a new emulator trained with simulator outputs with the designer
trained_emulator = designer.train(emulator, simulator)

# View the data the emulator is trained on
print("Training data:", trained_emulator.training_data, "\n")

# Make some predictions from the trained emulator:
print("Prediction at x = 0.1:", trained_emulator.predict(Input(0.1)))
print("Prediction at x = 0.2:", trained_emulator.predict(Input(0.2)))
