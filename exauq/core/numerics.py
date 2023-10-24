import math
from numbers import Real

FLOAT_TOLERANCE = 1e-9


def equal_to_tolerance(
    x: Real, y: Real, rel_tol: Real = FLOAT_TOLERANCE, abs_tol: Real = FLOAT_TOLERANCE
) -> bool:
    return math.isclose(x, y, rel_tol=rel_tol, abs_tol=abs_tol)
