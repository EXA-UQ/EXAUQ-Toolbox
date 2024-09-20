# Multi-Level Adaptive Sampling

This tutorial will show you how to extend an initial experimental design for a multi-level
Gaussian process (GP), using an adaptive sampling technique. Similarly to the
[non-levelled / single level case](./slas_tutorial.md), the idea is to take our current
multi-level GP and find a new design point (or batch of points) that will best
improve the fit of the multi-level GP. In contrast to the single level case, we also need
to determine the particular GP level for the new design point (or points), which
determines the simulator run(s) required.

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
f(x_1, x_2) = x_2 + x_1^2 + x_2^2 - \sqrt{2} + \mathrm{sin}(2\pi x_1) + \mathrm{sin}(4\pi x_1 x_2)
$$
with simulator domain defined as the rectangle $\mathcal{D}$ consisting of points
$(x_1, x_2)$ where $-1 \leq x_1 \leq 1$ and $1 \leq x_2 \leq 100$. We view this as the
top level of a 2-level multi-level simulator, with level 1 being given by the simpler
function
$$
f_1(x_1, x_2) = x_2 + x_1^2 + x_2^2 - \sqrt{2}
$$
and the difference function between the levels being
$$
\delta(x_1, x_2) = f(x_1, x_2) - f_1(x_1, x_2) = \mathrm{sin}(2\pi x_1) + \mathrm{sin}(4\pi x_1 x_2)
$$
As in the tutorial linked above, we will use a multi-level GP to emulate
$f = f_1 + \delta$.


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
using a Latin hypercube design (with the aid of [scipy](https://scipy.org/)) for creating
the training data for each level of the multi-level GP and defining the multi-level GP
to have Matern 5/2 kernel for each level. The full code for doing this is as follows:


``` { .python .copy }
from scipy.stats.qmc import LatinHypercube
from exauq.core.modelling import MultiLevel, TrainingDatum
from exauq.core.emulators import MogpEmulator
from exauq.core.modelling import MultiLevelGaussianProcess


# Set seed for repeatability.
sampler = LatinHypercube(domain.dim, seed=1)

# Create level 1 experimental design of 20 data points
lhs_inputs1 = [domain.scale(row) for row in sampler.random(n=20)]

# Create level 2 experimental design of 5 data points
lhs_inputs2 = [domain.scale(row) for row in sampler.random(n=5)]

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

<div class="result" markdown>
    Too few unique inputs; defaulting to flat priors
    Too few unique inputs; defaulting to flat priors
    Too few unique inputs; defaulting to flat priors
    Too few unique inputs; defaulting to flat priors
</div>


(The messages printed are from the `mogp_emulator` package and can be ignored: they
arise when initialising a new GP due to the face that it hasn't been trained on any data.)

## Extend the design using leave-one-out adaptive sampling (single new point)

Let's now find a new design point using the leave-one-out adaptive design methodology for
multi-level simulators / GPs. We use the function
[`compute_multi_level_loo_samples`][exauq.core.designers.compute_multi_level_loo_samples]
to do this. By default, a batch consisting of a single, new design point will be calculated. This function requires three arguments:

- The multi-level GP to find the new design point for.
- The [`SimulatorDomain`][exauq.core.modelling.SimulatorDomain] describing the domain on
  which the simulator is defined.
- The relative costs of running the simulator at each level. In our case, this should be
  the costs of running the level 1 simulator $f_1$ and the full level 2 simulator
  $f$ (not the difference $delta$).

Let's suppose for our toy example that the full simulator is 10-times more expensive to
run than the level 1 version, so that we assign a cost of 1 to $f_1$ and 10 to $f$.

In code, we perform the adaptive sampling as follows:


``` { .python .copy }
from exauq.core.designers import compute_multi_level_loo_samples

# Suppress warnings that arise from mogp_emulator
import warnings
warnings.filterwarnings("ignore")

costs = MultiLevel([1, 10])
level, new_design_pts = compute_multi_level_loo_samples(mlgp, domain, costs)

print("New design point:", new_design_pts[0])
print("Level to run it at:", level)
```

<div class="result" markdown>
    Prior solver failed to converge
</div>


<div class="result" markdown>
    New design point: (np.float64(-0.25057816255206145), np.float64(99.99999999957782))
    Level to run it at: 2
</div>


In order to update the fit of the multi-level GP, we need to know which level's GP needs
training, which we get from the `level` output by
[`compute_multi_level_loo_samples`][exauq.core.designers.compute_multi_level_loo_samples].
We then need to update that level GP with the appropriate training datum. In our case,
a level of 1 means we'd need to run the level 1 simulator $f_1$ on the new design point,
while a level of 2 means we need to run the **difference** $\delta$ of the full level 2
simulator and the level 1 simulator.

If instead we want to compute multiple new design points in one go, we can do this by
specifying a different batch size:


``` { .python .copy }
level, new_design_pts = compute_multi_level_loo_samples(mlgp, domain, costs, batch_size=5)

print("New design points:", new_design_pts)
print("Level to run batch at:", level)
```

<div class="result" markdown>
    Prior solver failed to converge
</div>


<div class="result" markdown>
    New design points: (Input(np.float64(-0.25057829462558634), np.float64(99.99999999801312)), Input(np.float64(-0.999999999976053), np.float64(70.97891960592197)), Input(np.float64(0.9999999999992959), np.float64(41.956772739235014)), Input(np.float64(0.5676976223902448), np.float64(1.0000000002884049)), Input(np.float64(0.11667965978007155), np.float64(99.99999999909942)))
    Level to run batch at: 2
</div>


Note that a single level is returned for the whole batch, i.e. all the new design points
should go into updating the GP at that level.

## Update the multi-level GP

The final step is to update the fit of the multi-level GP using the newly-calculated
design points and the given level. This first requires us to compute the correct simulator
outputs at the design points:


``` { .python .copy }
new_outputs = [ml_simulator[level](x) for x in new_design_pts]
new_data = [TrainingDatum(x, y) for x, y in zip(new_design_pts, new_outputs)]
```

To update the multi-level GP, we can fit the appropriate level GP with both the old and new training data for that level combined:


``` { .python .copy }
# mlgp[level].training_data is a tuple, so make a list to extend
# with the new data
data = list(mlgp[level].training_data) + new_data
mlgp[level].fit(data)

# Sense-check that the GP at the level has been updated
print(
    "Number of training data at each level:",
    {level: len(mlgp[level].training_data) for level in mlgp.levels}
    )
```

<div class="result" markdown>
    Number of training data at each level: {1: 20, 2: 10}
</div>


This completes one adaptive sampling 'iteration'. It's important to note that, when
creating a batch of multiple new design points, the fit of the GP is not updated between
the creation of each new point in the batch.

## Repeated application

In general, we can perform multiple adaptive sampling iterations to further improve the
fit of the multi-level GP with newly sampled design point(s). The following code goes
through five more sampling iterations, producing a single new design point at each
iteration. We have also introduced a helper function to assist in retraining the
multi-level GP. (Once again, the messages from the `mogp_emulator` package can be
ignored.)


``` { .python .copy }
def update_mlgp(mlgp, level, additional_data):
    """Updates the fit of the supplied multi-level GP with additional
    training data at a given level."""

    # mlgp[level].training_data is a tuple, so make a list to extend
    # with the new data
    data = list(mlgp[level].training_data) + additional_data
    mlgp[level].fit(data)


for i in range(5):
    # Find new design point adaptively (via batch of length 1)
    level, new_design_pts = compute_multi_level_loo_samples(mlgp, domain, costs)
    x = new_design_pts[0]

    # Compute simulator output at new design point
    y = ml_simulator[level](x)

    # Update GP fit
    new_data = [TrainingDatum(x, y)]
    update_mlgp(mlgp, level, new_data)

    # Print design point found and level applied at
    print(f"==> Updated level {level} with new design point {x}")

```

<div class="result" markdown>
    ==> Updated level 2 with new design point (np.float64(0.9999999999992757), np.float64(77.89999484785737))
</div>


<div class="result" markdown>
    ==> Updated level 2 with new design point (np.float64(0.5633828285924407), np.float64(78.88801714035827))
    Prior solver failed to converge
</div>


<div class="result" markdown>
    ==> Updated level 2 with new design point (np.float64(0.7044914999642888), np.float64(99.99999999477811))
</div>


<div class="result" markdown>
    ==> Updated level 2 with new design point (np.float64(0.9999999997618763), np.float64(23.04425024496773))
    Prior solver failed to converge
</div>


<div class="result" markdown>
    ==> Updated level 2 with new design point (np.float64(-0.9999999898496873), np.float64(87.19043644363062))
</div>
