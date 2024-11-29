import itertools
import math
import unittest

import numpy as np

from exauq.core.numerics import FLOAT_TOLERANCE, equal_within_tolerance, set_tolerance
from tests.utilities.utilities import exact, make_window


class TestEqualWithinTolerance(unittest.TestCase):
    def setUp(self) -> None:
        self.non_finite_values = [-math.inf, math.nan, math.inf]

    def assertAgreeOnRange(self, func1, func2, _range):
        for x in _range:
            self.assertIs(func1(x), func2(x))

    def test_equal_to_math_isclose_relative_tolerances(self):
        """Test that whether two reals are equal up to a relative tolerance agrees with
        the calculation given by math.isclose."""

        for x, rel_tol in itertools.product([-1, 1], [1e-1, 1e-2, 1e-3]):
            with self.subTest(x=x, rel_tol=rel_tol):
                # Note: use of abs_tol=0 forces the use of the relative tolerance
                self.assertAgreeOnRange(
                    lambda y: equal_within_tolerance(x, y, rel_tol=rel_tol, abs_tol=0),
                    lambda y: math.isclose(x, y, rel_tol=rel_tol, abs_tol=0),
                    _range=make_window(x, 2 * rel_tol, type="rel"),
                )

    def test_equal_to_math_isclose_absolute_tolerances(self):
        """Test that whether two reals are equal up to an absolute tolerance agrees with
        the calculation given by math.isclose."""

        for x, abs_tol in itertools.product([-0.1, 0, 0.1], [0.1, 0.05, 0.01]):
            with self.subTest(x=x, abs_tol=abs_tol):
                # Note: use of rel_tol=0 forces the use of the absolute tolerance
                self.assertAgreeOnRange(
                    lambda y: equal_within_tolerance(x, y, rel_tol=0, abs_tol=abs_tol),
                    lambda y: math.isclose(x, y, rel_tol=0, abs_tol=abs_tol),
                    _range=make_window(x, 2 * abs_tol),
                )

    def test_default_tolerances(self):
        """Test that the default tolerance used for both relative and absolute tolerances
        is equal to the package's float tolerance constant."""

        for x in [-1, 1]:
            with self.subTest(x=x):
                # Relative tolerance case
                self.assertAgreeOnRange(
                    lambda y: equal_within_tolerance(x, y),
                    lambda y: math.isclose(
                        x, y, rel_tol=FLOAT_TOLERANCE, abs_tol=FLOAT_TOLERANCE
                    ),
                    _range=make_window(x, 2 * FLOAT_TOLERANCE, type="rel"),
                )

                # Absolute tolerance case
                self.assertAgreeOnRange(
                    lambda y: equal_within_tolerance(x, y),
                    lambda y: math.isclose(
                        x, y, rel_tol=FLOAT_TOLERANCE, abs_tol=FLOAT_TOLERANCE
                    ),
                    _range=make_window(x, 2 * FLOAT_TOLERANCE),
                )

    def test_non_finite_values(self):
        """Test that infinite and NaN values are considered not equal to any finite
        number, no matter the tolerances used."""

        tolerances = [0, 1, 1e-9]
        for x, rel_tol, abs_tol in itertools.product(
            self.non_finite_values, tolerances, tolerances
        ):
            with self.subTest(x=x):
                self.assertFalse(
                    equal_within_tolerance(x, 1.1, rel_tol=rel_tol, abs_tol=abs_tol)
                )
                if math.isnan(x):
                    self.assertFalse(
                        equal_within_tolerance(x, x, rel_tol=rel_tol, abs_tol=abs_tol)
                    )
                else:
                    self.assertTrue(
                        equal_within_tolerance(x, x, rel_tol=rel_tol, abs_tol=abs_tol)
                    )

    def test_symmetric(self):
        """Test that the check for equality is symmetric in the args."""

        tolerances = [0, 1, 1e-9]
        values = self.non_finite_values + [-1, -0.001, 0, 0.99999, 1, 1.1]
        for x, rel_tol, abs_tol in itertools.product(values, tolerances, tolerances):
            with self.subTest(x=x):
                self.assertAgreeOnRange(
                    lambda y: equal_within_tolerance(
                        x, y, rel_tol=rel_tol, abs_tol=abs_tol
                    ),
                    lambda y: equal_within_tolerance(
                        y, x, rel_tol=rel_tol, abs_tol=abs_tol
                    ),
                    _range=values,
                )

    def test_negative_tolerances_error(self):
        """Test that a ValueError is raised if one of the tolerances supplied is
        negative."""

        with self.assertRaises(ValueError):
            equal_within_tolerance(1, 1, rel_tol=-0.01)

        with self.assertRaises(ValueError):
            equal_within_tolerance(1, 1, abs_tol=-0.01)

    def test_sequences(self):
        """Sequences are considered equal if they have the same length and each
        corresponding element is equal within the specified tolerance."""

        x = [0, 10]
        for tol in [1e-1, 1e-2, 1e-3]:
            # Compare with abs tolerance - should be determined by entry as index = 0
            with self.subTest(abs_tol=tol):
                for x0 in make_window(x[0], 2 * tol, type="abs"):
                    y = [x0, x[1]]
                    self.assertTrue(
                        equal_within_tolerance(x, y, rel_tol=0, abs_tol=tol)
                        is equal_within_tolerance(x[0], y[0], rel_tol=0, abs_tol=tol)
                    )

            # Compare with relative tolerance - should be determined by entry at index = 1
            with self.subTest(rel_tol=tol):
                for x1 in make_window(x[1], 2 * tol, type="rel"):
                    y = [x[0], x1]
                    self.assertTrue(
                        equal_within_tolerance(x, y, rel_tol=tol, abs_tol=0)
                        is equal_within_tolerance(x[1], y[1], rel_tol=tol, abs_tol=0)
                    )

    def test_numpy_arrays(self):
        """Numpy arrays can be compared like sequences."""

        x = np.array([0, 10])
        for tol in [1e-1, 1e-2, 1e-3]:
            # Compare with abs tolerance - should be determined by entry as index = 0
            with self.subTest(abs_tol=tol):
                for x0 in make_window(x[0], 2 * tol, type="abs"):
                    y = [x0, x[1]]
                    self.assertTrue(
                        equal_within_tolerance(x, y, rel_tol=0, abs_tol=tol)
                        is equal_within_tolerance(x[0], y[0], rel_tol=0, abs_tol=tol)
                    )

            # Compare with relative tolerance - should be determined by entry at index = 1
            with self.subTest(rel_tol=tol):
                for x1 in make_window(x[1], 2 * tol, type="rel"):
                    y = [x[0], x1]
                    self.assertTrue(
                        equal_within_tolerance(x, y, rel_tol=tol, abs_tol=0)
                        is equal_within_tolerance(x[1], y[1], rel_tol=tol, abs_tol=0)
                    )

    def test_nested_sequences_can_be_compared(self):
        """Sequences of sequences (or arrays of arrays etc) can be compared."""

        x = [[1, 2], [3, 4]]
        y = np.array([[1, 2], [3, 4]])
        self.assertTrue(equal_within_tolerance(x, y))

        y = (1, 2, 3, 4)
        self.assertFalse(equal_within_tolerance(x, y))


class TestSetTolerance(unittest.TestCase):
    def setup(self) -> None:
        self.tol = 1e-5

    def test_set_tolerance_tol_type_error(self):
        """This test ensures that a TypeError is raised if a float is not passed"""

        tol = [1, 2]
        with self.assertRaisesRegex(
            TypeError,
            exact(
                f"Expected 'tol' to be of type float, but receieved {type(tol)} instead."
            ),
        ):
            set_tolerance(tol)

    def test_set_tolerance_tol_negative_error(self):
        """This test ensures that a ValueError is raised if a non-positive float is passed"""

        tol = -1.5
        with self.assertRaisesRegex(
            ValueError, exact(f"Expected 'tol' to be non-negative but received {tol}.")
        ):
            set_tolerance(tol)

    def test_set_tolerance_check(self):
        """This test ensures the functionality of changing the global tolerance is working"""

        # Note: default tolerance is 1e-9 (hence should "fail")
        x = [1, 2]
        y = [1, 2 + 1e-6]
        assert not equal_within_tolerance(x, y)

        # If tolerance correctly changed - should now pass
        set_tolerance(1e-5)
        assert equal_within_tolerance(x, y)


if __name__ == "__main__":
    unittest.main()
