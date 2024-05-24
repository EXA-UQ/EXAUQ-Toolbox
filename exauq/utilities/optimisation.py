from numbers import Real
from typing import Callable, Optional

import numpy as np
import scipy.optimize
from numpy.typing import NDArray

from exauq.core.modelling import Input, SimulatorDomain
from exauq.core.numerics import FLOAT_TOLERANCE


def maximise(
    func: Callable[[NDArray], Real], domain: SimulatorDomain, seed: Optional[int] = None
) -> tuple[Input, float]:
    """Maximise an objective function over a simulator domain.

    Finds a point in the bounded input space defined by a simulator domain that maximises
    a given function, together with the maximum value. The underlying optimisation uses
    differential evolution, using the implementation in the Scipy package with the bounds
    that define the supplied simulator domain (see notes for further details.)

    The objective function `func` is expected to take a 1-dimensional Numpy array as an
    argument and to be defined for arrays corresponding to inputs from the given `domain`.
    (In other words, if ``x`` is for type ``Input`` and ``x in domain`` is ``true``, then
    ``func(numpy.array(x)))`` returns a finite real number.

    Parameters
    ----------
    func : Callable[[NDArray], numbers.Real]
        The objective function to maximise. Should take in a 1-dimensional Numpy array and
        return a real number.
    domain : SimulatorDomain
        The domain of a simulator, defining the bounded input space over which `func`
        will be maximised.
    seed : int, optional
        (Default: None) A number to seed the random number generator used in the
        underlying optimisation. If ``None`` then no seeding will be used.

    Returns
    -------
    tuple[Input, float]
        A pair ``(x, val)``, where ``x`` is the input from the domain that maximises the
        objective function and ``val`` is the maximum value of the objective function.

    Raises
    ------
    RuntimeError
        If finding the maximum value for the objective function failed for some reason
        (e.g. due to non-convergence).

    See Also
    --------
    The Scipy documentation for differential evolution optimisation:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html
    This is used with the `tol` and `atol` keyword arguments set to
    ``exauq.core.numerics.FLOAT_TOLERANCE``.

    Notes
    -----
    The optimisation is performed with a call to Scipy's
    ``scipy.optimize.differential_evolution`` function, with the bounds specified in
    ``domain.bounds``. The relative and absolute tolerances governing
    convergence (i.e. the ``tol`` and ``atol`` kwargs) are set to
    ``exauq.core.numerics.FLOAT_TOLERANCE``, and the ``seed`` parameter is exposed as
    described above, but otherwise the default kwargs are used in
    ``scipy.optimize.differential_evolution``.
    """

    if not isinstance(domain, SimulatorDomain):
        raise TypeError(
            f"Expected 'domain' to be of type SimulatorDomain, but received {type(domain)} instead."
        )

    try:
        y = func(np.array(domain.scale([0.5] * domain.dim)))
    except Exception:
        raise ValueError(
            "Expected 'func' to be a callable that takes a 1-dim Numpy array as argument."
        )

    if not isinstance(y, Real):
        raise ValueError(
            "Expected 'func' to be a callable that returns a real number, but instead "
            f"it returns type {type(y)}."
        )

    try:
        result = scipy.optimize.differential_evolution(
            lambda x: -func(x),
            bounds=domain.bounds,
            tol=FLOAT_TOLERANCE,
            atol=FLOAT_TOLERANCE,
            seed=seed,
        )
    except Exception as e:
        raise RuntimeError(f"Maximisation failed: {str(e)}")

    if not result.success:
        raise RuntimeError(f"Maximisation failed to converge: {result.message}")

    return Input(*result.x), -float(result.fun)
