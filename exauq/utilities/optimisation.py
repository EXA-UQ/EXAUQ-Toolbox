from typing import Callable

import numpy as np
import scipy.optimize
from numpy.typing import NDArray

from exauq.core.modelling import Input, SimulatorDomain
from exauq.core.numerics import FLOAT_TOLERANCE


def maximise(func: Callable[[NDArray], float], domain: SimulatorDomain) -> Input:
    if not isinstance(domain, SimulatorDomain):
        raise TypeError(
            f"Expected 'domain' to be of type SimulatorDomain, but received {type(domain)} instead."
        )

    try:
        _ = func(np.array(domain.scale([0.5] * domain.dim)))
    except Exception:
        raise ValueError(
            "Expected 'func' to be a callable that takes a 1-dim Numpy array as argument "
            "and returns a float."
        )

    result = scipy.optimize.differential_evolution(
        lambda x: -func(x),
        bounds=domain._bounds,
        tol=FLOAT_TOLERANCE,
        atol=FLOAT_TOLERANCE,
    )
    return Input(*result.x)
