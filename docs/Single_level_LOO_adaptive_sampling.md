# Tutorial: single level adaptive sampling

First we import the objects from the `exauq` package.


```python
from exauq.core.emulators import MogpEmulator  # A Gaussian process emulator backed by mogp-emulator
from exauq.core.modelling import (
    TrainingDatum,  # For working with emulator training data
    SimulatorDomain,  # For defining the input space of the emulator
)
from exauq.core.designers import compute_single_level_loo_samples  # Function for performing adaptive sampling

# Don't display warnings from mogp-emulator
import warnings
warnings.filterwarnings("ignore")
```

## Loading in training data

Read in some training data for the GP from a pre-prepared csv file:


```python
training_data = TrainingDatum.read_from_csv("./data/gp_training_data.csv", header = True)
training_data
```

## Train a GP

Next we'll train a GP. We first create a new GP using the `MogpEmulator` class, specifying that it uses a Matern 5/2 kernel.


```python
gp = MogpEmulator(kernel="Matern52")
```

Note that this GP hasn't been trained on any data yet; we can verify this be examining its `training_data` property:


```python
gp.training_data
```

Now we'll train it on the data we read in earlier.


```python
gp.fit(training_data)
```

## Find design point(s) using LOO adaptive sampling

Let's now find a new design point using the leave-one-out adaptive design methodology. We use the function `compute_single_level_loo_samples` to do this. This function requires two inputs:
- A GP to find the new design point for (which we have)
- A `SimulatorDomain` object, which defines the input space on which the simulator inputs are defined.

The training data points read in all have simulator inputs that lie in the unit square. To define the corresponding `SimulatorDomain`, we need to provide the lower and upper bounds for each of the coordinates. In this case, there are two input coordinates and each is bounded by 0 and 1.


```python
bounds = [(0, 1), (0, 1)]  # one pair of bounds for each input dimension
domain = SimulatorDomain(bounds)
```

As an aside, the `domain` object has a method for computing the 'pseudopoints' arising from a given set of simulator inputs. 'Pseudopoints' are used in the leave-one-out adaptive sampling methodology. To calculate the pseudopoints around the training inputs, we can do the following:


```python
# Get the training inputs
inputs = [datum.input for datum in training_data]

# Compute the pseudopoints
domain.calculate_pseudopoints(inputs)
```

Now we can generate a new design point:


```python
x = compute_single_level_loo_samples(gp, domain)
x
```

If instead we wanted to compute a batch of training inputs in one go, we can do this by specifying a different batch size:


```python
new_design_pts = compute_single_level_loo_samples(gp, domain, batch_size=5)
new_design_pts
```

Note how the new design points all lie within the simulator domain we defined earlier, i.e. they all lie in the unit square.

By default, the leave-one-out errors GP calculated during the adaptive sampling method uses a fresh copy of the supplied GP. In particular, it will use the same kernel function as the original. We can instead specify that a different GP is used by supplying a new one with the settings we desire. For example, to ensure that the leave-one-out errors GP uses a squared exponential kernel instead of a Matern 5/2, we can do the following:


```python
sqexp_gp = MogpEmulator(kernel="SquaredExponential")
x = compute_single_level_loo_samples(gp, domain, loo_errors_gp=sqexp_gp)
x
```
