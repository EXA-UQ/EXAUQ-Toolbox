# Multi-Level Adaptive Sampling

This tutorial will show you how to extend an initial experimental design for a multi-level
Gaussian process (GP), using an adaptive sampling technique. Similarly to the
[non-levelled / single level case](./slas_tutorial.md), the idea is to take our current
multi-level GP and find a new design point (or batch of points) that will best
improve the fit of the multi-level GP. In contrast to the single level case, we also need
to determine the particular GP level for the new design point (or points), which
determines the simulator run(s) required. To help us do this, the associated cost of running 
each level weighs the sampling criterion. 

This tutorial will show you how to:

* Use the function
  [`compute_multi_level_loo_samples`][exauq.core.designers.compute_multi_level_loo_samples]
  to extend an initial experimental design with a new design point (or batch of
  new design points), together with the level for the point(s).
* How to repeatedly add new design points using this method.

If you are unfamiliar with how to train a multi-level GP using the EXAUQ-Toolbox, you may
want to first work through the tutorial,
[Training a Multi-Level Gaussian Process Emulator](./training_multi_level_gp_tutorial.md). 
You may also wish to work through the
[Single Level Adaptive Sampling](./slas_tutorial.md) tutorial, to familiarise yourself
with adaptive sampling in the non-levelled case.

!!! note

    The function
    [`compute_multi_level_loo_samples`][exauq.core.designers.compute_multi_level_loo_samples]
    implements the cross-validation-based adaptive sampling for multi-level GPs described
    in Kimpton, L. M. et al. (2023) "Cross-Validation Based Adaptive Sampling for
    Multi-Level Gaussian Process Models". arXiv: <https://arxiv.org/abs/2307.09095>


## Setup

We'll work with the same toy simulator function found in the tutorial,
[Training a Multi-Level Gaussian Process Emulator](./training_multi_level_gp_tutorial.md). This is the function
$$
f_2(x_1, x_2) = x_2 + x_1^2 + x_2^2 - \sqrt{2} + \mathrm{sin}(2\pi x_1) + \mathrm{sin}(4\pi x_1 x_2)
$$
with simulator domain defined as the rectangle $\mathcal{D}$ consisting of points
$(x_1, x_2)$ where $0 \leq x_1 \leq 1$ and $-0.5 \leq x_2 \leq 1.5$. We view this as the
top level of a 2-level multi-level simulator, with level 1 being given by the simpler
function
$$
f_1(x_1, x_2) = x_2 + x_1^2 + x_2^2 - \sqrt{2}
$$
and the difference function between the levels being
$$
\delta(x_1, x_2) = f_2(x_1, x_2) - f_1(x_1, x_2) = \mathrm{sin}(2\pi x_1) + \mathrm{sin}(4\pi x_1 x_2)
$$
As in the tutorial linked above, we will use a multi-level GP to emulate
$f_2 = f_1 + \delta$ by fitting independent GPs.


We can express this in code as follows:


``` { .python .copy }
from exauq.core.modelling import SimulatorDomain, Input, MultiLevel
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

# Package up the level 1 simulator and delta into a single
# multi-level object. This makes the following code a bit
# nicer.
ml_simulator = MultiLevel([sim_level1, sim_delta])

```

## Initial design

To perform adaptive sampling, we need to begin with a multi-level GP trained with an initial design. We'll adopt the approach taken in
[Training a Multi-Level Gaussian Process Emulator](./training_multi_level_gp_tutorial.md),
using a Latin hypercube designer [`oneshot_lhs`][exauq.core.designers.oneshot_lhs] (with the aid of [scipy](https://scipy.org/)) for creating
the training data for each level of the multi-level GP and defining the multi-level GP
to have Matern 5/2 kernel for each level. The full code for doing this is as follows:


``` { .python .copy }
from exauq.core.designers import oneshot_lhs
from exauq.core.modelling import MultiLevel, TrainingDatum
from exauq.core.emulators import MogpEmulator
from exauq.core.modelling import MultiLevelGaussianProcess

# Create level 1 experimental design of 8 data points
lhs_inputs1 = oneshot_lhs(domain, 8, seed=1)

# Create level 2 experimental design of 4 data points
lhs_inputs2 = oneshot_lhs(domain, 4, seed=1)

# Put into a multi-level object
design = MultiLevel([lhs_inputs1, lhs_inputs2])

# Create outputs for each level (level 2 takes the delta between the two levels)
outputs1 = [sim_level1(x) for x in lhs_inputs1]
outputs2 = [sim_delta(x) for x in lhs_inputs2]

# Create training data
initial_data = MultiLevel([
    [TrainingDatum(x, y) for x, y in zip(lhs_inputs1, outputs1)],
    [TrainingDatum(x, y) for x, y in zip(lhs_inputs2, outputs2)],
])

# Define multi-level GP
gp1 = MogpEmulator(kernel="Matern52")
gp2 = MogpEmulator(kernel="Matern52")
mlgp = MultiLevelGaussianProcess([gp1, gp2])

# Fit to initial data
mlgp.fit(initial_data)
```

## Extend the design using leave-one-out adaptive sampling (single new point)

Let's now find a new design point using the leave-one-out adaptive design methodology for
multi-level simulators / GPs. We use the function
[`compute_multi_level_loo_samples`][exauq.core.designers.compute_multi_level_loo_samples]
to do this. By default, a batch consisting of a single, new design point will be calculated within a MultiLevel object. This function requires three arguments:

- The multi-level GP to find the new design point for.
- The [`SimulatorDomain`][exauq.core.modelling.SimulatorDomain] describing the domain on
  which the simulator is defined.
- The costs of running the simulator at each level. In our case, this should be
  the cost of running the level 1 simulator $f_1$ and then the additional cost for the full level 2 simulator
  $f$.

Let's suppose for our toy example that the full simulator is 10-times more expensive to
run than the level 1 version. This gives relative costs of 1 and 10 to $f_1$ and $f_2$ respectively.
Therefore, we assign a cost of 1 to $f_1$ and 11 to $f_2$ given the cost for our full simulator is $C_2 = c_1 + c_2$.

In code, we perform the adaptive sampling as follows:


``` { .python .copy }
from exauq.core.designers import compute_multi_level_loo_samples

# Suppress warnings that arise from mogp_emulator
import warnings
warnings.filterwarnings("ignore")

costs = MultiLevel([1, 11])
new_design_pt = compute_multi_level_loo_samples(mlgp, domain, costs)

level = list(new_design_pt.keys())[0]
new_input = new_design_pt.get(level)

print("New design point:", new_input)
print("Level to run it at:", level)
```

<div class="result" markdown>
    New design point: (Input(np.float64(-0.99999959132886), np.float64(63.30777655421669)),)
    Level to run it at: 2
    
</div>

In order to update the fit of the multi-level GP, we need to know which level's GP needs
training and the new input point which will be placed on that level. We get this from
[`compute_multi_level_loo_samples`][exauq.core.designers.compute_multi_level_loo_samples], 
which returns a MultiLevel object of new inputs. 

We then need to update that level GP with the appropriate training datum. In our case,
a level of 1 means we'd need to run the level 1 simulator $f_1$ on the new design point,
while a level of 2 means we need to run the **difference** $\delta$ of the full level 2
simulator and the level 1 simulator.

If instead we want to compute multiple new design points in one go, we can do this by
specifying a different batch size:


``` { .python .copy }
new_design_pts = compute_multi_level_loo_samples(mlgp, domain, costs, batch_size = 5)

for level in new_design_pts.levels:
    print(f"\nLevel: {level}")
    new_inputs = list(new_design_pts.get(level))
    print("New design points:", new_inputs)
```

<div class="result" markdown>
    
    Level: 1
    New design points: [Input(np.float64(0.9999999529517327), np.float64(60.30191266964135)), Input(np.float64(-0.4431057511688774), np.float64(99.99999831981881))]
    
    Level: 2
    New design points: [Input(np.float64(-0.9999999580156729), np.float64(63.31048199239925)), Input(np.float64(0.3477102541492165), np.float64(1.0000014551521161)), Input(np.float64(0.47336850801792996), np.float64(99.99999608324342))]
    
</div>

## Update the multi-level GP

The final step is to update the fit of the multi-level GP using the newly-calculated
MultiLevel design points from [`compute_multi_level_loo_samples`][exauq.core.designers.compute_multi_level_loo_samples]. For each level therefore we must:

1. Compute the correct simulator outputs at the new design points.
2. Create a list of TrainingDatum combining the inputs and outputs.
3. Add these into our new training data MultiLevel object. 


``` { .python .copy }
# Package the training data into a multi level object
new_training_data = MultiLevel({})

for level in new_design_pts.levels:

    # Take level inputs
    level_inputs = new_design_pts.get(level)

    # Run simulator for level outputs 
    level_outputs = [ml_simulator[level](x) for x in level_inputs]

    # Create TrainingDatum for each level
    level_training_datum = [TrainingDatum(x, y) for x, y in zip(level_inputs, level_outputs)]

    # Concatenate into full MultiLevel training data
    new_training_data = new_training_data + MultiLevel({level: level_training_datum})
```

To update the multi-level GP, we can then use the [`update`][exauq.core.modelling.AbstractGaussianProcess.update] method of mlgp by passing the newly created training data. 


``` { .python .copy }
mlgp.update(new_training_data)

# Sense-check that the GP at the level has been updated
print(
    "Number of training data at each level:",
    {level: len(mlgp[level].training_data) for level in mlgp.levels}
    )
```

<div class="result" markdown>
    Number of training data at each level: {1: 10, 2: 7}
    
</div>

This completes one adaptive sampling 'iteration'. It's important to note that, when
creating a batch of multiple new design points, the fit of the GP is not updated between
the creation of each new point in the batch.

## Repeated application

In general, we can perform multiple adaptive sampling iterations to further improve the
fit of the multi-level GP with newly sampled design point(s). The following code goes
through five more sampling iterations, producing a single new design point at each
iteration. We have also introduced a helper function to assist in retraining the
multi-level GP.


``` { .python .copy }
for i in range(5):
    # Find new design point adaptively (via batch of length 1)
    new_design_pt = compute_multi_level_loo_samples(mlgp, domain, costs)

    # Get level and input
    level = list(new_design_pt.keys())[0]
    x = new_design_pt.get(level)[0]

    # Compute simulator output at new design point
    y = ml_simulator[level](x)

    # Create MultiLevel TrainingDatum object
    training_data = MultiLevel({level: [TrainingDatum(x, y)]})
    
    # Update GP fit (Note passing output as a list)
    mlgp.update(training_data)

    # Print design point found and level applied at
    print(f"==> Updated level {level} with new design point {x}")

```

<div class="result" markdown>
    ==> Updated level 2 with new design point (np.float64(-0.4571354210550862), np.float64(99.99999774060194))
    
</div>

<div class="result" markdown>
    ==> Updated level 2 with new design point (np.float64(-0.9999999717831191), np.float64(29.973733429457244))
    
</div>

<div class="result" markdown>
    ==> Updated level 2 with new design point (np.float64(-0.7536919941567507), np.float64(99.99999997690348))
    
</div>

<div class="result" markdown>
    ==> Updated level 2 with new design point (np.float64(-0.2245432598199002), np.float64(99.9999999954828))
    
</div>

<div class="result" markdown>
    ==> Updated level 2 with new design point (np.float64(-0.11433834585452296), np.float64(32.92075037083332))
    
