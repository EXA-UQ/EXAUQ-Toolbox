"""
Contains the tolerance checks that are required within numerics calculations with the 
`FLOAT_TOLERANCE` attribute alongside the ability to set your own global tolerance. 


Tolerance Control
---------------------------------------------------------------------------------------
[`FLOAT_TOLERANCE`][exauq.core.numerics.FLOAT_TOLERANCE]
Global attribute of tolerance for toolbox

[`equal_within_tolerance`][exauq.core.numerics.equal_within_tolerance]
Function to check equality of two real numbers up to a tolerance

[`set_tolerance`][exauq.core.numerics.set_tolerance]
Function used to set global tolerance


"""

import math
from collections.abc import Sequence
from numbers import Real
from typing import Union

import numpy as np

FLOAT_TOLERANCE = 1e-9
"""The default tolerance to use when testing for equality of real numbers."""


def equal_within_tolerance(
    x: Union[Real, Sequence[Real]],
    y: Union[Real, Sequence[Real]],
    rel_tol: Real = None,
    abs_tol: Real = None,
) -> bool:
    """Test equality of two real numbers or sequences of real numbers up to a tolerance.

    This function compares either two real numbers or two sequences of real numbers
    element-wise, determining if they are close within specified tolerances.

    Parameters
    ----------
    x, y :
        Real numbers or sequences of real numbers to test equality of.
    rel_tol :
        The maximum allowed relative difference. Dynamically passed at runtime to allow
        for updating of FLOAT_TOLERANCE. Defaults to FLOAT_TOLERANCE (1e-9) if no change.
    abs_tol :
        The minimum permitted absolute difference. Dynamically passed at runtime to allow
        for updating of FLOAT_TOLERANCE. Defaults to FLOAT_TOLERANCE (1e-9) if no change.

    Returns
    -------
    bool
        Whether the two numbers or sequences of numbers are equal up to the relative and absolute tolerances.
    """
    # Use FLOAT_TOLERANCE if rel_tol and abs_tol aren't provided
    rel_tol = rel_tol or FLOAT_TOLERANCE
    abs_tol = abs_tol or FLOAT_TOLERANCE

    if _is_seq(x) and _is_seq(y):
        return len(x) == len(y) and all(
            equal_within_tolerance(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
            for a, b in zip(x, y)
        )
    elif isinstance(x, Real) and isinstance(y, Real):
        return math.isclose(x, y, rel_tol=rel_tol, abs_tol=abs_tol)
    else:
        raise TypeError(
            f"Both 'arguments' must be of type {Real}, type sequences or type Numpy arrays, "
            "but one or more arguments were of an unexpected type."
        )


def _is_seq(x) -> bool:
    return isinstance(x, (Sequence, np.ndarray))


def set_tolerance(tol: float):
    """
    Allows the updating of the global FLOAT_TOLERANCE from default (1e-9) to the tol value
    passed.

    Parameters
    ----------
    tol :
        The new tolerance you wish to set the global FLOAT_TOLERANCE to.
    """

    if not isinstance(tol, float):
        raise TypeError(
            f"Expected 'tol' to be of type float, but received {type(tol)} instead."
        )

    if tol < 0:
        raise ValueError(f"Expected 'tol' to be non-negative but received {tol}.")

    global FLOAT_TOLERANCE
    FLOAT_TOLERANCE = tol
