# Training a Multi-Level Gaussian Process Emulator

The purpose of this tutorial is to demonstrate how to train a multi-level Gaussian process
(GP) to emulate a simulator. It uses the same example simulator from the tutorial
[Training a Gaussian Process Emulator](./training_gp_tutorial.md), which demonstrates
training a GP in the classical, non-levelled paradigm; you may wish to work through that
tutorial first if you haven't done so already.

This tutorial will show you how to:

* Work with multi-level objects (such as training data) using the
  [`MultiLevel`][exauq.core.modelling.MultiLevel] class.
* Create training data for a multi-level emulation scenario.
* Define and train a multi-level Gaussian process.
* Make new predictions of simulator outputs using the trained multi-level GP.

!!! note
    
    Due to the pseudo-stochastic nature of the algorithms for fitting
    Gaussian processes, you may get slight differences in some of the code outputs in
    this tutorial.

## A toy multi-level simulator

This tutorial will look at taking a multi-level approach to emulating the toy simulator
found in the tutorial,
[Training A Gaussian Process Emulator](./training_gp_tutorial.md).
This is is defined to be the mathematical function
$$
f(x_1, x_2) = x_2 + x_1^2 + x_2^2 - \sqrt{2} + \mathrm{sin}(2\pi x_1) \
+ \mathrm{sin}(4\pi x_1 x_2)
$$
defined on the rectangular domain $\mathcal{D}$ consisting of 2d points
$(x_1, x_2)$, where $-1 \leq x_1 \leq 1$ and $1 \leq x_2 \leq 100$.

We will consider this as the top level of a multi-level simulator of two levels. The level
1 version is the simpler function
$$
f_1(x_1, x_2) = x_2 + x_1^2 + x_2^2 - \sqrt{2}
$$
In the multi-level paradigm, the idea is to emulate the whole simulator $f$ with a
multi-level GP, in this case having two levels. The multi-level GP is
itself essentially a sum of two GPs, one at each level. The GP at the first level emulates
the level 1 function $f_1$, while the second level GP emulates the **difference** between
the second and first level simulators:
$$
\delta(x_1, x_2) = f(x_1, x_2) - f_1(x_1, x_2) = \mathrm{sin}(2\pi x_1) + \mathrm{sin}(4\pi x_1 x_2)
$$
The reason for taking the difference is so that the full simulator $f$ is the sum of $f_1$
and $\delta$, which allows us to emulate each function separately via a multi-level GP.

We express all this in code as follows:


``` { .python .copy }
from exauq.core.modelling import SimulatorDomain, Input
import numpy as np

# The bounds define the lower and upper bounds on each coordinate
domain = SimulatorDomain(bounds=[(-1, 1), (1, 100)])

# The full simulator (at level 2)
def sim_func(x: Input) -> float:
    return (
        x[1] + x[0]**2 + x[1]**2 - np.sqrt(2)
        + np.sin(2 * np.pi * x[0]) + np.sin(4 * np.pi * x[0] * x[1])
    )

# The level 1 simulator
def sim_level1(x: Input) -> float:
    return x[1] + x[0]**2 + x[1]**2 - np.sqrt(2)

# The difference between levels 1 and 2
def sim_delta(x: Input) -> float:
    return sim_func(x) - sim_level1(x)
```

## Multi-level objects

In order to help structure objects in the multi-level paradigm, the EXAUQ-Toolbox provides
the [`MultiLevel`][exauq.core.modelling.MultiLevel] class. This is like a dictionary,
except that the keys are integers representing the levels. For example, we can create a
multi-level collection of floating point numbers, for 3 levels, like so:


``` { .python .copy }
from exauq.core.modelling import MultiLevel

# Creates floating point numbers at levels 1, 2 and 3
ml_numbers = MultiLevel([1.1, 2.2, 3.3])

# Get the numbers for each level using dictionary access notation []
print("Level 1 value:", ml_numbers[1])
print("Level 2 value:", ml_numbers[2])
print("Level 3 value:", ml_numbers[3])
```

<div class="result" markdown>
    Level 1 value: 1.1
    Level 2 value: 2.2
    Level 3 value: 3.3
</div>


In general, providing a sequence of length `n` to
[`MultiLevel`][exauq.core.modelling.MultiLevel] will assign the list elements to the
levels `1, 2, ..., n` in order.

We can get the levels in a multi-level collection by using the
[`levels`][exauq.core.modelling.MultiLevel.levels] property:


``` { .python .copy }
ml_numbers.levels
```




<div class="result" markdown>
    (1, 2, 3)
</div>



As an application, let's use the [`MultiLevel`][exauq.core.modelling.MultiLevel] class to
encapsulate the different levels of our simulator, which will make our code a little
neater later:


``` { .python .copy }
ml_simulator = MultiLevel([sim_level1, sim_delta])
```

## Creating multi-level training data

The setup for doing multi-level emulation is similar to the
[single level case](./training_gp_tutorial.md), with the exception that we work with
multi-level objects. We need to construct some multi-level training data, utilising
experimental designs for each level's simulator, then train a multi-level GP with this
data.

To create the training data, we'll use a Latin hypercube design (with the aid of
[scipy](https://scipy.org/)) at each level. (For more detailed explanation of creating an
experimental design from a Latin hypercube sample, see the section,
**Creating an experimental design** from the
[Training a Gaussian Process Emulator](./training_gp_tutorial.md) tutorial.)


``` { .python .copy }
from scipy.stats.qmc import LatinHypercube
from exauq.core.modelling import MultiLevel, TrainingDatum

# Set seed for repeatability.
sampler = LatinHypercube(domain.dim, seed=1)

# Create level 1 experimental design of 20 data points
lhs_array1 = sampler.random(n=20)
lhs_inputs1 = [domain.scale(row) for row in lhs_array1]

# Create level 2 experimental design of 5 data points
lhs_array2 = sampler.random(n=5)
lhs_inputs2 = [domain.scale(row) for row in lhs_array2]

# Put into a multi-level object
design = MultiLevel([lhs_inputs1, lhs_inputs2])

```

Next, we calculate the simulator outputs and create the training data, doing this for
each level separately. Note how we use the multi-level object of simulator functions we
created earlier.


``` { .python .copy }
# Create level 1 simulator outputs and training data
outputs1 = [ml_simulator[1](x) for x in design[1]]
data1 = [TrainingDatum(x, y) for x, y in zip(design[1], outputs1)]

# Create level 2 simulator outputs and training data
outputs2 = [ml_simulator[2](x) for x in design[2]]
data2 = [TrainingDatum(x, y) for x, y in zip(design[2], outputs2)]

# Combine into a multi-level object
training_data = MultiLevel([data1, data2])
```

If we wish, we can verify that we have the correct data at each level by doing some
manual inspections:


``` { .python .copy }
print("Number of level 1 training data:", len(training_data[1]))
print("Number of level 2 training data:", len(training_data[2]))

# Show the first couple of data points for each level:
print("\nLevel 1:")
print(repr(training_data[1][0]))
print(repr(training_data[1][1]))

print("\nLevel 2:")
print(repr(training_data[2][0]))
print(repr(training_data[2][1]))
```

<div class="result" markdown>
    Number of level 1 training data: 20
    Number of level 2 training data: 5
    
    Level 1:
    TrainingDatum(input=Input(np.float64(-0.45118216247002574), np.float64(75.49520470318662)), output=np.float64(5773.8104896605955))
    TrainingDatum(input=Input(np.float64(0.18558403872803675), np.float64(6.204185236670643)), output=np.float64(43.31632756065012))
    
    Level 2:
    TrainingDatum(input=Input(np.float64(-0.6860872668651894), np.float64(8.14123867468156)), output=np.float64(0.04053108865750232))
    TrainingDatum(input=Input(np.float64(0.2779780667419962), np.float64(61.11931671766958)), output=np.float64(0.857129851613081))
</div>


## Defining and fitting a multi-level GP

Next, we need to define a multi-level GP with two levels, which we can do using the
[`MultiLevelGaussianProcess`][exauq.core.modelling.MultiLevelGaussianProcess] class.
To construct it, we need to create GPs for each level, which we'll do using the
[`MogpEmulator`][exauq.core.emulators.MogpEmulator] class, with a Matern 5/2 kernel for
each level.


``` { .python .copy }
from exauq.core.emulators import MogpEmulator
from exauq.core.modelling import MultiLevelGaussianProcess

gp1 = MogpEmulator(kernel="Matern52")
gp2 = MogpEmulator(kernel="Matern52")

mlgp = MultiLevelGaussianProcess([gp1, gp2])
```

<div class="result" markdown>
    Too few unique inputs; defaulting to flat priors
    Too few unique inputs; defaulting to flat priors
    Too few unique inputs; defaulting to flat priors
    Too few unique inputs; defaulting to flat priors
</div>


(The messages printed are from the `mogp_emulator` package and can be ignored: they
arise when initialising a new GP due to the fact that it hasn't been trained on any data.)

As with ordinary GPs, we can verify that our multi-level GP hasn't yet been trained on
data. Note that each level of the GP has its own training data, so the
[`training_data`][exauq.core.modelling.MultiLevelGaussianProcess.training_data]
property of the multi-level GP is a [`MultiLevel`][exauq.core.modelling.MultiLevel] object: 


``` { .python .copy }
mlgp.training_data
```




<div class="result" markdown>
    MultiLevel({1: (), 2: ()})
</div>



Finally, we train the multi-level GP with the multi-level data we created earlier, using
the [`fit`][exauq.core.modelling.MultiLevelGaussianProcess.fit] method:


``` { .python .copy }
mlgp.fit(training_data)

# Verify that the data is as we expect
assert len(mlgp.training_data[1]) == 20
assert len(mlgp.training_data[2]) == 5
```

## Making predictions with the multi-level GP

To finish off, let's use our newly-trained multi-level GP to estimate the output of our
top-level simulator at a new input. We make a prediction with the multi-level GP using the
[`predict`][exauq.core.modelling.MultiLevelGaussianProcess.predict] method. As described
in the tutorial, [Training a Gaussian Process Emulator](./training_gp_tutorial.md), the
prediction consists of both the point estimate and a measure of the uncertainty of the
prediction:


``` { .python .copy }
x = Input(0.5, 50)
prediction = mlgp.predict(x)

print(prediction)
print("Point estimate:", prediction.estimate)
print("Variance of estimate:", prediction.variance)
print("Standard deviation of estimate:", prediction.standard_deviation)
```

<div class="result" markdown>
    GaussianProcessPrediction(estimate=np.float64(2549.8545301180034), variance=np.float64(1.236147336654947), standard_deviation=1.1118216298736714)
    Point estimate: 2549.8545301180034
    Variance of estimate: 1.236147336654947
    Standard deviation of estimate: 1.1118216298736714
</div>


Let's see how well the prediction did against the true simulator value:


``` { .python .copy }
y = sim_func(x)  # the true value
pct_error = 100 * abs((prediction.estimate - y) / y)

print("Predicted value:", prediction.estimate)
print("Actual simulator value:", y)
print("Percentage error:", pct_error)
```

<div class="result" markdown>
    Predicted value: 2549.8545301180034
    Actual simulator value: 2548.835786437627
    Percentage error: 0.03996898057524886
</div>


As in the non-levelled case, we can also calculate the normalised expected square error
for the prediction:


``` { .python .copy }
prediction.nes_error(y)
```




<div class="result" markdown>
    0.7947014462360164
</div>


