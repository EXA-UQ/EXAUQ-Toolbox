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
    Input(np.float64(0.30013627275662463), np.float64(1.4999999988571904))
</div>



If instead we want to compute multiple new design points in one go, we can do this by
specifying a different batch size:


``` { .python .copy }
new_design_pts = compute_single_level_loo_samples(gp, domain, batch_size=5)
new_design_pts
```




<div class="result" markdown>
    (Input(np.float64(0.30007824221629625), np.float64(1.4999999979529648)),
     Input(np.float64(0.999999994587522), np.float64(0.6826546284669834)),
     Input(np.float64(0.9999999986244122), np.float64(-0.17596770942664763)),
     Input(np.float64(5.107706646523269e-09), np.float64(-0.09235535068408796)),
     Input(np.float64(0.5334868922322674), np.float64(0.6071752586381185)))
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

new_outputs
```




<div class="result" markdown>
    [np.float64(2.790146409960019),
     np.float64(1.483283040738709),
     np.float64(-1.3610261842057274),
     np.float64(-1.4980393760923938),
     np.float64(-1.1635894082593787)]
</div>



Then to update the GP, we create a list of TrainingDatum to pass into the [`update`][exauq.core.modelling.AbstractGaussianProcess] method. 


``` { .python .copy }
training_data = [TrainingDatum(x, y) for x, y in zip(new_design_pts, new_outputs)]
gp.update(training_data)

# Sense-check that the updated GP now has the combined data
print("Number of training data:", len(gp.training_data))
```

<div class="result" markdown>
    Number of training data: 13
    
</div>

This completes one adaptive sampling 'iteration'. It's important to note that, when
creating multiple new design points in a batch, the fit of the GP is **not** updated between
the creation of each new point in the batch.

## Repeated application

In general, we can perform multiple adaptive sampling iterations to further improve the
fit of the GP with newly sampled design point(s). The following code goes through five
more sampling iterations, producing a single new design point at each iteration.


``` { .python .copy }
for i in range(5):
    # Find new design point adaptively (via batch of length 1)
    x = compute_single_level_loo_samples(gp, domain)[0]
    
    # Compute simulator output at new design point
    y = sim_func(x)

    # Create TrainingDatum "list"
    training_data = [TrainingDatum(x, y)]

    # Update GP fit 
    gp.update(training_data)

    # Print design point found and level applied at
    print(f"==> Updated with new design point {x}")

```

<div class="result" markdown>
    ==> Updated with new design point (np.float64(0.0836011108112355), np.float64(1.499992781886386))
    
</div>

<div class="result" markdown>
    ==> Updated with new design point (np.float64(0.8108464748068719), np.float64(-0.4651335698390211))
    
</div>

<div class="result" markdown>
    ==> Updated with new design point (np.float64(0.19709840570930104), np.float64(-0.4999826354426524))
    
</div>

<div class="result" markdown>
    ==> Updated with new design point (np.float64(0.6428329298517965), np.float64(0.14819712595125878))
    
</div>

<div class="result" markdown>
    ==> Updated with new design point (np.float64(1.3533000489840408e-05), np.float64(1.185357521648959))
    
