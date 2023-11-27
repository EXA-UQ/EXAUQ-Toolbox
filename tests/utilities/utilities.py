"""Functions etc. to support testing"""

from numbers import Real
from typing import Literal, Optional, Tuple

import numpy as np
from exauq.core.modelling import Input
from exauq.core.numerics import equal_within_tolerance


def exact(string: str):
    """Turn a string into a regular expressions that defines an exact match on the
    string.
    """
    escaped = string
    for char in ["\\", "(", ")", "[", "]"]:
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


def compare_input_tuples(tuple1: Tuple[Input, ...], tuple2: Tuple[Input, ...]) -> bool:
    """
    Compares two tuples of Input objects, considering each object's value and accounting for duplicates.

    This function checks if each Input object in `tuple1` has a corresponding Input object in `tuple2` with a similar
    value, within a specified tolerance. The comparison is order-independent and ensures that duplicates are matched
    correctly. If any Input object in `tuple1` does not have a corresponding match in `tuple2`, or if the tuples are
    of different lengths, the function returns False.

    Args:
        tuple1 (Tuple[Input, ...]): The first tuple of Input objects.
        tuple2 (Tuple[Input, ...]): The second tuple of Input objects.

    Returns: bool: True if the tuples are equivalent (each element in `tuple1` has a matching element in `tuple2`
    within tolerance), False otherwise.
    """

    if len(tuple1) != len(tuple2):
        return False

    used_indices = set()

    for item1 in tuple1:
        match_found = False
        for i, item2 in enumerate(tuple2):
            if i not in used_indices and equal_within_tolerance(item1.value, item2.value):
                used_indices.add(i)
                match_found = True
                break

        if not match_found:
            return False

    return True
