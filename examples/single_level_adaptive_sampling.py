# A quick-and-dirty implementation of cross-validation-based single level
# adaptive sampling experimental design, using mogp-emulator

import math
import numpy as np
import mogp_emulator


def f(x1: float, x2: float) -> float:
    """The simulator being emulated"""
    if not 0 <= x1 <= 1 and 0 <= x2 <= 1:
        raise ValueError("ERROR: input variable(s) out of range")

    return x2 + x1**2 + x2**2 - math.sqrt(2)


def es_loo(gp, theta, experiments, observations, i):
    # Refit GP at all points except the ith one (using the given parameters)
    pass


if __name__ == "__main__":
    
    # Create initial design
    one_shot_designer = mogp_emulator.LatinHypercubeDesign(2, (0, 1))
    n_initial_samples = 7  # number of samples for initial design
    initial_experiments = one_shot_designer.sample(n_initial_samples)
    initial_observations = np.array([f(x1, x2) for (x1, x2) in initial_experiments])
    
    print("Initial experiments:\n", initial_experiments, "\n")
    print("Initial observations:\n", initial_observations, "\n")
    
    # Initialise a GP using a Matern kernel
    gp = mogp_emulator.GaussianProcess(initial_experiments, initial_observations, kernel='Matern52')

    # Fit GP to the initial data
    gp = mogp_emulator.fit_GP_MAP(gp)

    print("Hyperparameters:", gp.theta)

    # Get mean and variance for distribution at a new input
    x = np.array([.3, .3])
    mean, variance, _ = gp.predict(x)
    print("mean at x =", mean)
    print("variance at x =", variance)

    n_adaptive_iterations = 5
    for iteration in range(n_adaptive_iterations):
        pass


