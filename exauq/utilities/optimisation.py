import random

from numbers import Real
from typing import Callable, Optional

import scipy.optimize

import exauq.core.numerics as numerics
from exauq.core.modelling import Input, SimulatorDomain

# Maximum seed determined by scipy seeding
MAX_SEED = 2**32 - 1


def maximise(
    func: Callable[[Input], Real], domain: SimulatorDomain, seed: Optional[int] = None
) -> tuple[Input, float]:
    """Maximise an objective function over a simulator domain.

    Finds a point in the bounded input space defined by a simulator domain that maximises
    a given function, together with the maximum value. The underlying optimisation uses
    differential evolution, using the implementation in the Scipy package with the bounds
    that define the supplied simulator domain (see notes for further details.)

    Parameters
    ----------
    func : Callable[[Input], numbers.Real]
        The objective function to maximise. Should take an Input object from the given
        `domain` as an argument and return a real number.
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

    if seed is not None and not isinstance(seed, int):
        raise TypeError(
            f"Random seed must be an integer, but received type {type(seed)}."
        )

    try:
        y = func(domain.scale([0.5] * domain.dim))
    except Exception:
        raise ValueError(
            "Expected 'func' to be a callable that takes an argument of type "
            f"{Input.__name__}."
        )

    if not isinstance(y, Real):
        raise ValueError(
            "Expected 'func' to be a callable that returns a real number, but instead "
            f"it returns type {type(y)}."
        )

    try:
        result = scipy.optimize.differential_evolution(
            lambda x: -func(Input.from_array(x)),
            bounds=domain.bounds,
            tol=numerics.FLOAT_TOLERANCE,
            atol=numerics.FLOAT_TOLERANCE,
            seed=seed,
        )
    except Exception as e:
        raise RuntimeError(f"Maximisation failed: {str(e)}")

    if not result.success:
        raise RuntimeError(f"Maximisation failed to converge: {result.message}")

    return Input(*result.x), -float(result.fun)


def generate_seeds(seed: int | None, batch_size: int) -> tuple:
    """
    Generate a tuple of unique seeds from an initial seed equal to the length of the batch_size
    passed by `compute_single_level_loo_samples` or `compute_multilevel_loo_samples`

    MAX_SEED is set as the seed cap due to the scipy seeding system within `scipy.optimize` 
    using legacy 32 bit integers. Crucially however, the seeds are there for reproducibility rather 
    than being used as part of the generation of LOO samples.

    Parameters
    ----------
    seed :
        The initial seed to seed the random sample of seeds generated. If None is passed
        will simply return a tuple of None equal to length of batch_size.
    batch_size:
        The length of the array of seeds generated

    Returns
    --------
    tuple
        A tuple of seeds generated of length batch_size and seeded from the initial seed.
    """
    if seed is None:
        return tuple([None] * batch_size)

    elif not isinstance(seed, int):
        raise TypeError(
            f"Expected 'seed' to be None or of type int, but received {type(seed)} instead."
        )

    if not isinstance(batch_size, int):
        raise TypeError(
            f"Expected 'batch_size' to be of type int, but received {type(batch_size)} instead."
        )

    if seed < 0 and not None:
        raise ValueError(
            f"Expected 'seed' to be None or >=0, but received {seed} instead."
        )

    if batch_size < 1 or batch_size >= MAX_SEED:
        raise ValueError(
            f"Expected 'batch_size' to be >=1 and <{MAX_SEED}, but received {batch_size} instead."
        )

    random.seed(seed)
    seeds = random.sample(range(0, MAX_SEED), batch_size)

    return tuple(map(int, seeds))
