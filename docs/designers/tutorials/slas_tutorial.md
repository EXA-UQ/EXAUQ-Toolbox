# Single Level Adaptive Sampling

This tutorial will show you how to extend an initial experimental design for a Gaussian
process (GP) emulator, using an adaptive sampling technique. The idea is to take our
current GP and sequentially find new design points (or batches of points) that will have 'the greatest
impact' on improving the fit of the GP.

This tutorial will show you how to:

* Use the function
  [`compute_single_level_loo_samples`][exauq.core.designers.compute_single_level_loo_samples]
  to extend an initial experimental design with a new design point (or batch of
  new design points).
* How to repeatedly add new design points using this method.

If you are unfamiliar with how to train a GP using the EXAUQ-Toolbox, you may want to
first work through the tutorial, [Training A Gaussian Process Emulator](./training_gp_tutorial.md).

!!! note

    The function
    [`compute_single_level_loo_samples`][exauq.core.designers.compute_single_level_loo_samples]
    implements the cross-validation-based adaptive sampling for GPs, as described in
    Mohammadi, H. et al. (2022) "Cross-Validation-based Adaptive Sampling for Gaussian process models". DOI:
    <https://doi.org/10.1137/21M1404260>.


## Setup

We'll work with the same toy simulator function found in the tutorial,
[Training A Gaussian Process Emulator](./training_gp_tutorial.md). This is the function
$$
f(x_1, x_2) = x_2 + x_1^2 + x_2^2 - \sqrt{2} + \mathrm{sin}(2\pi x_1) + \mathrm{sin}(4\pi x_1 x_2)
$$
with simulator domain defined as the rectanglular input space $\mathcal{D}$ consisting of points
$(x_1, x_2)$ where $0 \leq x_1 \leq 1$ and $-0.5 \leq x_2 \leq 1.5$. We can express this in
code as follows:


``` { .python .copy }
from exauq.core.modelling import SimulatorDomain, Input
import numpy as np

# The bounds define the lower and upper bounds on each coordinate
domain = SimulatorDomain(bounds=[(0, 1), (-0.5, 1.5)])

def sim_func(x: Input) -> float:
    return (
        x[1] + x[0]**2 + x[1]**2 - np.sqrt(2)
        + np.sin(2 * np.pi * x[0]) + np.sin(4 * np.pi * x[0] * x[1])
    )
```

## Initial design

To perform adaptive sampling, we need to begin with a Gaussian process (GP) emulator
trained with an initial design. We'll do this by using a Latin hypercube designer [`oneshot_lhs`][exauq.core.designers.oneshot_lhs] (with the
aid of [scipy](https://scipy.org/)) and using a GP with a Matern 5/2 kernel. The approach below is a condensed version of that found in the tutorial
[Training A Gaussian Process Emulator](./training_gp_tutorial.md).


``` { .python .copy }
from exauq.core.designers import oneshot_lhs
from exauq.core.modelling import TrainingDatum
from exauq.core.emulators import MogpEmulator

# Create Latin hypercube sample, setting a seed to make the sampling repeatable.
lhs_inputs = oneshot_lhs(domain, 8, seed=1)

# Calculate simulator outputs, using our toy simulator function.
outputs = [sim_func(x) for x in lhs_inputs]

# Create the training data of input/output pairs.
data = [TrainingDatum(x, y) for x, y in zip(lhs_inputs, outputs)]

# Define a GP with a Matern 5/2 kernel and fit to the data.
gp = MogpEmulator(kernel="Matern52")
gp.fit(data)
```

## Extend the design using leave-one-out adaptive sampling

Let's now find a new design point using the leave-one-out adaptive design methodology. The
idea is to take our current GP and find a new design point (or batch of points) that will
have 'the greatest impact' on improving the fit of the GP, when combined with the
corresponding simulator output (or outputs in the batch case). We use the function
[`compute_single_level_loo_samples`][exauq.core.designers.compute_single_level_loo_samples]
to do this. This function requires two arguments:

- The GP to find the new design point for.
- The [`SimulatorDomain`][exauq.core.modelling.SimulatorDomain] describing the domain on
  which the simulator is defined.

By default, a batch consisting of a single, new design point will be calculated:


``` { .python .copy }
from exauq.core.designers import compute_single_level_loo_samples

# Suppress warnings that arise from mogp_emulator
import warnings
warnings.filterwarnings("ignore")

new_design_pts = compute_single_level_loo_samples(gp, domain)
new_design_pts[0]
```




<div class="result" markdown>
    Input(np.float64(0.30007640211188036), np.float64(1.4999999862795013))
</div>



If instead we want to compute multiple new design points in one go, we can do this by
specifying a different batch size:


``` { .python .copy }
new_design_pts = compute_single_level_loo_samples(gp, domain, batch_size=5)
new_design_pts
```




<div class="result" markdown>
    (Input(np.float64(0.3000973111062418), np.float64(1.4999999663266008)),
     Input(np.float64(0.9999999963281517), np.float64(0.6827426542564956)),
     Input(np.float64(0.9999999912029516), np.float64(-0.17591079544032318)),
     Input(np.float64(2.4334379733481626e-08), np.float64(-0.09229890650081729)),
     Input(np.float64(1.6255335522075143e-08), np.float64(1.1989655466639983)))
</div>



Note how the new design points all lie within the simulator domain we defined earlier,
i.e. they all lie in the rectanglar input space $\mathcal{D}$.

It's worth pointing out that these design points are not equal to any of the training inputs
for the GP:


``` { .python .copy }
training_inputs = [datum.input for datum in gp.training_data]
for x in new_design_pts:
    assert not any(x == x_train for x_train in training_inputs)
```

## Update the GP

The final step is to update the fit of the GP using the newly-calculated design points.
This first requires us to compute the simulator values at the design points (in our case,
using the toy function defined earlier) in order to create new training data:


``` { .python .copy }
new_outputs = [sim_func(x) for x in new_design_pts]
new_data = [TrainingDatum(x, y) for x, y in zip(new_design_pts, new_outputs)]
```

To update the GP, we fit it with both the old and new training data combined:


``` { .python .copy }
# gp.training_data is a tuple, so make a list to extend
# with the new data
data = list(gp.training_data) + new_data
gp.fit(data)

# Sense-check that the updated GP now has the combined data
print("Number of training data:", len(gp.training_data))
```

<div class="result" markdown>
    Number of training data: 13
    
</div>

This completes one adaptive sampling 'iteration'. It's important to note that, when
creating multiple new design points in a batch, the fit of the GP is not updated between
the creation of each new point in the batch.

## Repeated application

In general, we can perform multiple adaptive sampling iterations to further improve the
fit of the GP with newly sampled design point(s). The following code goes through five
more sampling iterations, producing a single new design point at each iteration. We have
also introduced a helper function to assist in retraining the GP. (Once again, the
messages from the `mogp_emulator` package can be ignored.)


``` { .python .copy }
def update_gp(gp, additional_data):
    """Updates the fit of the supplied GP with additional training data."""

    # gp.training_data is a tuple, so make a list to extend
    # with the new data
    data = list(gp.training_data) + additional_data
    gp.fit(data)


for i in range(5):
    # Find new design point adaptively (via batch of length 1)
    x = compute_single_level_loo_samples(gp, domain)[0]
    
    # Compute simulator output at new design point
    y = sim_func(x)

    # Update GP fit
    new_data = [TrainingDatum(x, y)]
    update_gp(gp, new_data)

    # Print design point found and level applied at
    print(f"==> Updated with new design point {x}")

```

<div class="result" markdown>
    ==> Updated with new design point (np.float64(0.7497466279972856), np.float64(1.4697059369662921))
    
</div>

<div class="result" markdown>
    ==> Updated with new design point (np.float64(0.1157312460738138), np.float64(-0.3475394033707414))
    
</div>

<div class="result" markdown>
    ==> Updated with new design point (np.float64(0.5236521661973336), np.float64(1.399928492998442))
    
</div>

<div class="result" markdown>
    ==> Updated with new design point (np.float64(0.7092931215153352), np.float64(0.7742519356263287))
    
</div>

<div class="result" markdown>
    ==> Updated with new design point (np.float64(0.7890211417416498), np.float64(-0.49998726677696703))
    
