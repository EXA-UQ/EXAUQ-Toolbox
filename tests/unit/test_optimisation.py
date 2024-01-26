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
            return -float(x[0] + np.sqrt(2) * np.sin(x[0]))

        domain = SimulatorDomain([(2, 100)])
        x = maximise(f, domain)
        argmax = Input(5 * np.pi / 4)
        self.assertEqualWithinTolerance(argmax, x, rel_tol=1e-5)

    def test_function_arg_errors(self):
        """A ValueError is raised if the provided function does not accept Numpy arrays as
        args or does not return a real number. A TypeError is raised the supplied domain
        is not of type SimulatorDomain."""

        domain = SimulatorDomain([(0, 1)])

        # Does not accept Numpy arrays as args
        def f(x: dict) -> float:
            return x["a"]

        with self.assertRaisesRegex(
            ValueError,
            exact(
                "Expected 'func' to be a callable that takes a 1-dim Numpy array as argument."
            ),
        ):
            _ = maximise(f, domain)

        # Returns non-real objects
        def f(x: np.ndarray):
            return np.array([np.sum(x)])

        return_type = type(f(np.array([1])))

        with self.assertRaisesRegex(
            ValueError,
            exact(
                "Expected 'func' to be a callable that returns a real number, but instead "
                f"it returns type {return_type}."
            ),
        ):
            _ = maximise(f, domain)

    def test_domain_type_error(self):
        """A TypeError is raised the supplied domain is not of type SimulatorDomain."""

        arg = "a"
        with self.assertRaisesRegex(
            TypeError,
            exact(
                f"Expected 'domain' to be of type SimulatorDomain, but received {type(arg)} instead."
            ),
        ):
            _ = maximise(np.sum, domain=arg)

    def test_failed_convergence_error(self):
        """A RuntimeError is raised if convergence failed in the maximisation."""

        domain = SimulatorDomain([(0, 1)])

        def f(x):
            return x[0] if x[0] < 0.5 else float("inf")

        with self.assertRaisesRegex(
            RuntimeError,
            "^Maximisation failed to converge: ",
        ):
            _ = maximise(f, domain)


if __name__ == "__main__":
    unittest.main()
