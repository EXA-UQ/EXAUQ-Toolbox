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
This is defined to be the mathematical function
$$
f_2(x_1, x_2) = x_2 + x_1^2 + x_2^2 - \sqrt{2} + \mathrm{sin}(2\pi x_1) \
+ \mathrm{sin}(4\pi x_1 x_2)
$$
defined on the rectangular domain $\mathcal{D}$ consisting of 2d points
$(x_1, x_2)$, where $0 \leq x_1 \leq 1$ and $-0.5 \leq x_2 \leq 1.5$.

We will consider this as the top level of a multi-level simulator of two levels. The level
1 version is a simpler function, which is typically cheaper to run than the top level simulator. 
$$
f_1(x_1, x_2) = x_2 + x_1^2 + x_2^2 - \sqrt{2} \\
$$
In the multi-level paradigm, the idea is to emulate the top level simulator $f_2$ with a
multi-level GP, in this case having two levels. The multi-level GP is
a sum of two GPs, one at each level. The GP at the first level emulates
the level 1 function $f_1$, while the second level GP emulates the **difference** between
the second and first level simulators:
$$
\delta(x_1, x_2) = f_2(x_1, x_2) - f_1(x_1, x_2) = \mathrm{sin}(2\pi x_1) + \mathrm{sin}(4\pi x_1 x_2)
$$
We take this delta because the difference between the levels is often a simpler function than the top level itself.

!!! note

    The multi-level GP approach that we are taking is an autoregressive approach to decompose the top level into the sum of the lower level 
    plus the difference between the two levels as described in Kennedy, Marc & O'Hagan, A. (1998) "Predicting the Output from a Complex Computer Code When Fast Approximations Are Available". 
    DOI:<https://doi.org/10.1093/biomet/87.1.1>

We express all this in code as follows:


``` { .python .copy }
from exauq.core.modelling import SimulatorDomain, Input
import numpy as np

# The bounds define the lower and upper bounds on each coordinate
domain = SimulatorDomain(bounds=[(0, 1), (-0.5, 1.5)])

# The full simulator (at level 2)
def sim_level2(x: Input) -> float:
    return (
        x[1] + x[0]**2 + x[1]**2 - np.sqrt(2)
        + np.sin(2 * np.pi * x[0]) + np.sin(4 * np.pi * x[0] * x[1])
    )

# The level 1 simulator
def sim_level1(x: Input) -> float:
    return x[1] + x[0]**2 + x[1]**2 - np.sqrt(2)

# The difference between levels 1 and 2
def sim_delta(x: Input) -> float:
    return sim_level2(x) - sim_level1(x)
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



One other useful function of [MultiLevel][exauq.core.modelling.MultiLevel] is the ability to concatenate different MultiLevel objects together into one single MultiLevel object using the [+][exauq.core.modelling.MultiLevel.__add__] operator.


``` { .python .copy }
# Create Multilevels by passing dictionaries with singular list values
mlevel1 = MultiLevel({1: [1, 2, 3]})
mlevel2 = MultiLevel({1: [4, 5], 2: [1, 2, 3]})

# Concatenate the MultiLevel objects together 
mlevel3 = mlevel1 + mlevel2

mlevel3
```




<div class="result" markdown>
    MultiLevel({1: (1, 2, 3, 4, 5), 2: (1, 2, 3)})
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

To create the training data, we'll use a Latin hypercube designer [`oneshot_lhs`][exauq.core.designers.oneshot_lhs] (with the aid of
[scipy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.LatinHypercube.html)) at each level. (For more detailed explanation of creating an
experimental design from a Latin hypercube sample, see the section,
**Creating an experimental design** from the
[Training a Gaussian Process Emulator](./training_gp_tutorial.md) tutorial.)


``` { .python .copy }
from exauq.core.designers import oneshot_lhs
from exauq.core.modelling import MultiLevel, TrainingDatum

# Create level 1 experimental design of 20 data points
lhs_inputs1 = oneshot_lhs(domain, 20, seed=1)

# Create level 2 experimental design of 5 data points
lhs_inputs2 = oneshot_lhs(domain, 5, seed=1)

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

!!! note
    
    It is worth noting here for clarity that delta is calculated by `lhs_inputs2` being run through both level 1 and level 2 (see `sim_delta` function) to  make a single training point for delta. 
    These points run through level 1 for calculating this delta are **not** automatically included in the training data for the level 1 GP.  


If we wish, we can verify that we have the correct data at each level by doing some
manual inspections:


``` { .python .copy }
print("Number of level 1 training data:", len(training_data[1]))
print("Number of level 2 training data:", len(training_data[2]))

# Show the first couple of data points for each level:
print("\nLevel 1:")
TrainingDatum.tabulate(training_data[1], rows = 2)

print("\nLevel 2:")
TrainingDatum.tabulate(training_data[2], rows = 2)
```

<div class="result" markdown>
    Number of level 1 training data: 20
    Number of level 2 training data: 5
    
    Level 1:
    Inputs:                                 Output:             
    ------------------------------------------------------------
    0.2744089188        1.0049536304        0.6759721219        
    0.5927920194        -0.3948649447       -1.3017578043       
    
    Level 2:
    Inputs:                                 Output:             
    ------------------------------------------------------------
    0.2976356751        1.1198145215        0.0897465476        
    0.7711680775        0.7205402211        -0.3473985727       
    
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

As with single-level GPs, we can verify that our multi-level GP hasn't yet been trained on
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
x = Input(0.5, 1.5)
prediction = mlgp.predict(x)

print(prediction)
print("Point estimate:", prediction.estimate)
print("Variance of estimate:", prediction.variance)
print("Standard deviation of estimate:", prediction.standard_deviation)
```

<div class="result" markdown>
    GaussianProcessPrediction(estimate=np.float64(2.428045966540852), variance=np.float64(0.4494867879424077), standard_deviation=0.6704377584402654)
    Point estimate: 2.428045966540852
    Variance of estimate: 0.4494867879424077
    Standard deviation of estimate: 0.6704377584402654
    
</div>

Let's see how well the prediction did against the true simulator value:


``` { .python .copy }
y = sim_level2(x)  # the true value
pct_error = 100 * abs((prediction.estimate - y) / y)

print("Predicted value:", prediction.estimate)
print("Actual simulator value:", y)
print("Percentage error:", pct_error)
```

<div class="result" markdown>
    Predicted value: 2.428045966540852
    Actual simulator value: 2.5857864376269055
    Percentage error: 6.1002899849230925
    
</div>

As in the non-levelled case, we can also calculate the normalised expected square error
for the prediction:


``` { .python .copy }
prediction.nes_error(y)
```




<div class="result" markdown>
    0.708081529557582
</div>


