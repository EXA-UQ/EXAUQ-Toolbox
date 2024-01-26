import unittest

import numpy as np

from exauq.core.modelling import Input, SimulatorDomain
from exauq.utilities.optimisation import maximise
from tests.utilities.utilities import ExauqTestCase, exact


class TestMaximise(ExauqTestCase):
    def setUp(self) -> None:
        self.domain = SimulatorDomain([(0, 1)])
        self.seed = 1

    def f(self, x: np.ndarray) -> float:
        return float(np.sin(1 / x)) if float(x) > 0 else 0

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
        x = maximise(f, domain, seed=self.seed)
        argmax = Input(5 * np.pi / 4)
        self.assertEqualWithinTolerance(argmax, x, rel_tol=1e-5)

    def test_unseeded_by_default(self):
        """The maximisation is not seeded by default, as evidenced by repeated
        runs with the same args giving (slightly) different results."""

        x1 = maximise(self.f, self.domain)
        x2 = maximise(self.f, self.domain)

        # Take .value to return a float and test for exact inequality
        self.assertNotEqual(x1.value, x2.value)

    def test_repeated_results_when_seed_set(self):
        """The output of the maximisation is the same when the seed is the same."""

        x1 = maximise(self.f, self.domain, seed=self.seed)
        x2 = maximise(self.f, self.domain, seed=self.seed)

        # Take .value to return a float and test for exact equality
        self.assertEqual(x1.value, x2.value)

    def test_function_arg_errors(self):
        """A ValueError is raised if the provided function does not accept Numpy arrays as
        args or does not return a real number. A TypeError is raised the supplied domain
        is not of type SimulatorDomain."""

        # Does not accept Numpy arrays as args
        def f(x: dict) -> float:
            return x["a"]

        with self.assertRaisesRegex(
            ValueError,
            exact(
                "Expected 'func' to be a callable that takes a 1-dim Numpy array as argument."
            ),
        ):
            _ = maximise(f, self.domain)

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
            _ = maximise(f, self.domain)

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

    @unittest.skip("to be fixed in upcoming PR")
    def test_failed_convergence_error(self):
        """A RuntimeError is raised if convergence failed in the maximisation."""

        def f(x):
            return x[0] if x[0] < 0.5 else float("inf")

        with self.assertRaisesRegex(
            RuntimeError,
            "^Maximisation failed to converge: ",
        ):
            _ = maximise(f, self.domain, seed=self.seed)


if __name__ == "__main__":
    unittest.main()
