import math
from numbers import Real

FLOAT_TOLERANCE = 1e-9
"""The default tolerance to use when testing for equality of real numbers."""


def equal_within_tolerance(
    x: Real, y: Real, rel_tol: Real = FLOAT_TOLERANCE, abs_tol: Real = FLOAT_TOLERANCE
) -> bool:
    """Test equality of two real numbers up to a tolerance.

    This is a thin wrapper around the ``isclose`` function from the standard library
    ``math`` module, behaving in exactly the same way. The meanings of the parameters are
    the same as given in that function; the only difference from ``isclose`` is that the
    `rel_tol` and `abs_tol` parameters are set to the value of ``FLOAT_TOLERANCE`` by
    default. See the documentation for ``isclose`` for more detail.

    Parameters
    ----------
    x, y : numbers.Real
        Real numbers to test equality of.
    rel_tol : numbers.Real, optional
        (Default: ``FLOAT_TOLERANCE``) The maximum allowed difference permitted relative
        to the larger of the absolute values of `x` and `y`.
    abs_tol : numbers.Real, optional
        (Default: ``FLOAT_TOLERANCE``) The minimum permitted absolute difference between
        `x` and `y`.

    Returns
    -------
    bool
        Whether the two numbers are equal up to the relative and absolute tolerances.

    See Also
    --------

    [math.isclose](https://docs.python.org/3/library/math.html#math.isclose) : Standard library function

    """

    return math.isclose(x, y, rel_tol=rel_tol, abs_tol=abs_tol)
