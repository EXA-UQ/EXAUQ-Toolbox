# Let's suppose we start with some simulations from a multi-level simulator, with two
# levels. We'll assume the simulators operate on a 3-dimensional input space, with
# each coordinate bounded by 0 and 1.
from exauq.core.modelling import Input, MultiLevel, SimulatorDomain, TrainingDatum

domain = SimulatorDomain(bounds=[(0, 1), (0, 1), (0, 1)])

data_level1 = [
    TrainingDatum(Input(0.109, 0.122, 0.491), -0.806),
    TrainingDatum(Input(0.998, 0.69, 0.217), 0.993),
    TrainingDatum(Input(0.269, 0.813, 0.034), 0.067),
    TrainingDatum(Input(0.115, 0.281, 0.108), -0.353),
    TrainingDatum(Input(0.521, 0.428, 0.213), -0.513),
    TrainingDatum(Input(0.123, 0.624, 0.496), -0.92),
    TrainingDatum(Input(0.749, 0.335, 0.653), 0.768),
    TrainingDatum(Input(0.735, 0.327, 0.039), 0.231),
    TrainingDatum(Input(0.982, 0.557, 0.682), 1.454),
    TrainingDatum(Input(0.075, 0.818, 0.218), -0.422),
]
data_level2 = [
    TrainingDatum(Input(0.332, 0.766, 0.306), 0.38),
    TrainingDatum(Input(0.84, 0.906, 0.791), -0.065),
    TrainingDatum(Input(0.477, 0.125, 0.482), -0.237),
    TrainingDatum(Input(0.92, 0.27, 0.235), 1.346),
    TrainingDatum(Input(0.414, 0.909, 0.977), 0.138),
]
data = MultiLevel([data_level1, data_level2])

# Next, let's assume that the simulator at level 2 takes 20 times longer to run than
# the simulator at level 1.

costs = MultiLevel([1, 20])

# Also, we assume that the level 2 simulator is completely (positively) correlated with
# the level 1 simulator. This is actually the default assumption, but we can be explicit
# if we like:
correlations = MultiLevel([1])  # Note: only corresponds to level 1

# The idea now is to initialise a multi-level GP on the successive differences across
# levels of the simulator outputs. To do this, we need to supply an 'empty' multi-level
# GP, which will be fit to this data. We will use a GP with Matern 5/2 kernel and zero
# mean.
from exauq.core.emulators import MogpEmulator
from exauq.core.modelling import MultiLevelGaussianProcess

gp_level1 = MogpEmulator(kernel="Matern52")
gp_level2 = MogpEmulator(kernel="Matern52")
mlgp = MultiLevelGaussianProcess(
    [gp_level1, gp_level2]
)  # Note: set all coefficients to 1

# To then fit this multi-level GP to the data comprising of inter-level differences, and
# also compute the costs of running the simulators to calculate new output differences, we
# call the following function:
from exauq.core.designers import initialise_for_multi_level_loo_sampling

mlgp, delta_costs = initialise_for_multi_level_loo_sampling(
    mlgp, data, costs, correlations
)

# Now we can use this prepared multi-level GP and the associated simulator costs to compute
# a batch of two new samples:
from exauq.core.designers import compute_multi_level_loo_samples

new_design_pts = compute_multi_level_loo_samples(mlgp, domain, delta_costs, batch_size=2)

print(new_design_pts)
