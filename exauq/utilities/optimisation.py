from typing import Callable

import numpy as np
import scipy.optimize
from numpy.typing import NDArray

from exauq.core.modelling import Input, SimulatorDomain
from exauq.core.numerics import FLOAT_TOLERANCE


def maximise(func: Callable[[NDArray], float], domain: SimulatorDomain) -> Input:
    """Maximise an objective function over a simulator domain.

    Finds a point in the bounded input space defined by a simulator domain that maximses
    a given function. The underlying optimisation uses differential evolution, as
    implemented in the Scipy package, with the relative and absolute tolerances governing
    convergence being set to ``exauq.core.numerics.FLOAT_TOLERANCE``.

    The objective function `func` is expected to take a 1-dimensional Numpy array as an
    argument and to be defined for arrays corresponding to inputs from the given `domain`.
    (In other words, if ``x`` is for type ``Input`` and ``x in domain`` is ``true``, then
    ``func(numpy.array(x)))`` returns a finite floating point number.

    Parameters
    ----------
    func : Callable[[NDArray], float]
        The objective function to maximse. Should take in a 1-dimensional Numpy array and
        return a float.
    domain : SimulatorDomain
        The domain of a simulator, defining the bounded input space over which `func`
        will be maximsed.

    Returns
    -------
    Input
        The point in the domain that maximises the objective function, as an ``Input``.

    See Also
    --------

    The Scipy documentation for differential evolution optimisation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html
    This is used with the `tol` and `atol` keyword arguments set to
    ``exauq.core.numerics.FLOAT_TOLERANCE``.
    """
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
        bounds=domain.bounds,
        tol=FLOAT_TOLERANCE,
        atol=FLOAT_TOLERANCE,
    )
    return Input(*result.x)
