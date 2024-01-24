from typing import Callable

import scipy.optimize
from numpy.typing import NDArray

from exauq.core.modelling import Input, SimulatorDomain
from exauq.core.numerics import FLOAT_TOLERANCE


def maximise(func: Callable[[NDArray], float], domain: SimulatorDomain) -> Input:
    result = scipy.optimize.differential_evolution(
        lambda x: -func(x),
        bounds=domain._bounds,
        tol=FLOAT_TOLERANCE,
        atol=FLOAT_TOLERANCE,
    )
    return Input(*result.x)
