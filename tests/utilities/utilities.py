"""Functions etc. to support testing"""

import unittest
from numbers import Real
from typing import Literal, Optional, Tuple

import numpy as np

from exauq.core.modelling import Input
from exauq.core.numerics import FLOAT_TOLERANCE, equal_within_tolerance


class ExauqTestCase(unittest.TestCase):
    """A subclass of ``unittest.TestCase`` with some extra assertions useful for testing
    the `exauq` package."""

    def assertEqualWithinTolerance(
        self, x1, x2, rel_tol=FLOAT_TOLERANCE, abs_tol=FLOAT_TOLERANCE
    ) -> None:
        """Test for equality using the `numerics.equal_within_tolerance` function.

        Note that this does *not* check that the two arguments `x1` and `x2` have the same
        type. So, for example, a list and a Numpy array containing the same real number
        values will be considered equal."""

        self.assertTrue(
            equal_within_tolerance(x1, x2, rel_tol=rel_tol, abs_tol=abs_tol),
            msg=f"assertEqualWithinTolerance: Values {x1} and {x2} not equal within tolerance.",
        )
        return None

    def assertNotEqualWithinTolerance(
        self, x1, x2, rel_tol=FLOAT_TOLERANCE, abs_tol=FLOAT_TOLERANCE
    ) -> None:
        """Test for inequality using the `numerics.equal_within_tolerance` function.

        Note that this does *not* check that the two arguments `x1` and `x2` have the same
        type. So, for example, a list and a Numpy array containing the same real number
        values will be considered equal."""

        self.assertFalse(
            equal_within_tolerance(x1, x2, rel_tol=rel_tol, abs_tol=abs_tol),
            msg=f"assertEqualWithinTolerance: Values {x1} and {x2} equal within tolerance.",
        )
        return None

    def assertArraysEqual(self, arr1, arr2) -> None:
        self.assertIsInstance(arr1, np.ndarray, "'arr1' is not a Numpy array")
        self.assertIsInstance(arr2, np.ndarray, "'arr2' is not a Numpy array")
        self.assertTrue(
            np.array_equal(arr1, arr2, equal_nan=True), "arrays are not equal"
        )

    def assertArraysNotEqual(self, arr1, arr2) -> None:
        self.assertIsInstance(arr1, np.ndarray, "'arr1' is not a Numpy array")
        self.assertIsInstance(arr2, np.ndarray, "'arr2' is not a Numpy array")
        self.assertFalse(np.array_equal(arr1, arr2, equal_nan=True), "arrays are equal")


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
