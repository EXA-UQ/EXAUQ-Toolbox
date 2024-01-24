import unittest

import numpy as np

from exauq.core.modelling import Input, SimulatorDomain
from exauq.utilities.optimisation import maximise
from tests.utilities.utilities import ExauqTestCase, exact


class TestMaximise(ExauqTestCase):
    def test_constraints_obeyed(self):
        """The input returned lies in the supplied domain."""

        def negative_sum_squares(x: np.ndarray) -> float:
            return -np.sum(x**2)

        domains = [
            SimulatorDomain([(-1, 1)]),
            SimulatorDomain([(0, 1), (1, 2)]),
            SimulatorDomain([(0, 1), (1, 2), (-1, 1)]),
        ]
        for domain in domains:
            with self.subTest(domain=domain):
                x = maximise(negative_sum_squares, domain)
                self.assertTrue(x in domain)

    def test_maximises_globally(self):
        """The input returned maximises the supplied function on the whole domain,
        not just locally."""

        # Following function has lots of local maxima
        def f(x: np.ndarray) -> float:
            return -float(x + np.sqrt(2) * np.sin(x))

        domain = SimulatorDomain([(2, 100)])
        x = maximise(f, domain)
        argmax = Input(5 * np.pi / 4)
        self.assertEqualWithinTolerance(argmax, x, rel_tol=1e-5)

    def test_arg_errors(self):
        """A ValueError is raised if the provided function does not accept Numpy arrays as
        args. A TypeError is raised the supplied domain is not of type SimulatorDomain."""
        domain = SimulatorDomain([(0, 1)])

        def f(x: dict) -> float:
            return x["a"]

        with self.assertRaisesRegex(
            ValueError,
            exact(
                "Expected 'func' to be a callable that takes a 1-dim Numpy array as argument "
                "and returns a float."
            ),
        ):
            _ = maximise(f, domain)

        arg = "a"
        with self.assertRaisesRegex(
            TypeError,
            exact(
                f"Expected 'domain' to be of type SimulatorDomain, but received {type(arg)} instead."
            ),
        ):
            _ = maximise(np.sum, domain=arg)


if __name__ == "__main__":
    unittest.main()
