import math
from collections.abc import Sequence
from numbers import Real
from typing import Union

FLOAT_TOLERANCE = 1e-9
"""The default tolerance to use when testing for equality of real numbers."""


# TODO: replace with more careful version when this becomes available
def equal_within_tolerance(
    x: Union[Real, Sequence[Real]],
    y: Union[Real, Sequence[Real]],
    rel_tol: Real = FLOAT_TOLERANCE,
    abs_tol: Real = FLOAT_TOLERANCE,
) -> bool:
    """Test equality of two real numbers or sequences of real numbers up to a tolerance.

    This function compares either two real numbers or two sequences of real numbers
    element-wise, determining if they are close within specified tolerances.

    Parameters
    ----------
    x, y : Union[numbers.Real, Sequence[numbers.Real]]
        Real numbers or sequences of real numbers to test equality of.
    rel_tol : numbers.Real, optional
        The maximum allowed relative difference.
    abs_tol : numbers.Real, optional
        The minimum permitted absolute difference.

    Returns
    -------
    bool
        Whether the two numbers or sequences of numbers are equal up to the relative and absolute tolerances.
    """
    if isinstance(x, Sequence) and isinstance(y, Sequence):
        if len(x) != len(y):
            return False
        return all(
            math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol) for a, b in zip(x, y)
        )
    elif isinstance(x, Real) and isinstance(y, Real):
        return math.isclose(x, y, rel_tol=rel_tol, abs_tol=abs_tol)
    else:
        raise TypeError("Both arguments must be either numbers or sequences of numbers.")
