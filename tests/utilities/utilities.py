"""Functions etc. to support testing"""

from numbers import Real
from typing import Literal, Optional

import numpy as np


def exact(string: str):
    """Turn a string into a regular expressions that defines an exact match on the
    string.
    """
    escaped = string
    for char in ["\\", "(", ")"]:
        escaped = escaped.replace(char, _escape(char))

    return "^" + escaped + "$"


def _escape(char):
    return "\\" + char


def make_window(
    x: Real, tol: float, type: Optional[Literal["abs", "rel"]] = None, num: int = 50
):
    """Make a range of numbers around a number with boundaries defined by a tolerance.

    This function is useful for generating ranges of numbers that will form part of a
    test for numerical equality up to some tolerance.

    If `type` is equal to ``'abs'``, then the range returned will be `num` equally-spaced
    numbers between ``x - tol`` and ``x + tol``. If `type` is ``'rel'``, then the range
    will be `num` (linearly) equally-spaced numbers between ``(1 - tol) * x`` and
    ``x / (1 - tol)``. If `type` is equal to ``None`` then the type will be set to
    ``"abs"`` if ``abs(x) < tol`` or to ``"rel"`` otherwise.
    """

    if type is None:
        _type = "abs" if abs(x) < tol else "rel"
        return make_window(x, tol, type=_type, num=num)
    if type == "abs":
        return np.linspace(x - tol, x + tol, num=num)
    elif type == "rel":
        return np.linspace(x * (1 - tol), x / (1 - tol), num=num)
    else:
        raise ValueError("'type' must equal one of 'abs' or 'rel'")


def compare_input_tuples(tuple1, tuple2):
    """
    Compares two tuples of Input objects by converting them to tuples of their values.

    Args:
    tuple1: The first tuple of Input objects.
    tuple2: The second tuple of Input objects.

    Returns:
    True if the tuples contain the same Input values (regardless of order), False otherwise.
    """
    set1 = set(input_obj.value for input_obj in tuple1)
    set2 = set(input_obj.value for input_obj in tuple2)

    return set1 == set2
