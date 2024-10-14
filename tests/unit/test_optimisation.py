import unittest

import numpy as np

from exauq.core.modelling import Input, SimulatorDomain
from exauq.utilities.optimisation import maximise, generate_seeds
from tests.utilities.utilities import ExauqTestCase, exact


class TestMaximise(ExauqTestCase):
    def setUp(self) -> None:
        self.domain = SimulatorDomain([(0, 1)])
        self.seed = 1

    def f(self, x: Input) -> float:
        return np.sin(1 / x[0]) if x[0] > 0 else 0

    def test_constraints_obeyed(self):
        """The input returned lies in the supplied domain."""

        def negative_sum_squares(x: Input) -> float:
            return -sum([z**2 for z in x])

        domains = [
            SimulatorDomain([(-1, 1)]),
            SimulatorDomain([(0, 1), (1, 2)]),
            SimulatorDomain([(0, 1), (1, 2), (-1, 1)]),
        ]
        for domain in domains:
            with self.subTest(domain=domain):
                x, _ = maximise(negative_sum_squares, domain)
                self.assertTrue(x in domain)

    def test_maximum_value_returned(self):
        """The maximum value of the supplied function is given along with the input
        where this maximum is achieved."""

        def f(x: Input) -> float:
            return sum(x)

        domain = SimulatorDomain([(0, 1), (0, 1), (0, 1)])
        x, max_val = maximise(f, domain, seed=self.seed)
        self.assertEqualWithinTolerance(3, max_val, rel_tol=1e-5)
        self.assertEqualWithinTolerance(max_val, f(x))

    def test_maximises_globally(self):
        """The input returned maximises the supplied function on the whole domain,
        not just locally."""

        # Following function has lots of local maxima
        def f(x: Input) -> float:
            return -float(x[0] + np.sqrt(2) * np.sin(x[0]))

        domain = SimulatorDomain([(2, 100)])
        x, _ = maximise(f, domain, seed=self.seed)
        argmax = Input(5 * np.pi / 4)
        self.assertEqualWithinTolerance(argmax, x, rel_tol=1e-5)

    def test_unseeded_by_default(self):
        """The maximisation is not seeded by default, as evidenced by repeated
        runs with the same args giving (slightly) different results."""

        # Take .value to return a float and test for exact inequality
        results = {maximise(self.f, self.domain)[0].value for _ in range(10)}

        # We expect more than one unique result
        self.assertGreater(
            len(results),
            1,
            "Expected more than one unique result",
        )

    def test_non_integer_seed_error(self):
        """A TypeError is raised if the provided seed is not an integer (or None)."""

        seed = 1.1
        with self.assertRaisesRegex(
            TypeError,
            exact(f"Random seed must be an integer, but received type {type(seed)}."),
        ):
            _ = maximise(self.f, self.domain, seed=seed)

    def test_repeated_results_when_seed_set(self):
        """The output of the maximisation is the same when the seed is the same."""

        x1, _ = maximise(self.f, self.domain, seed=self.seed)
        x2, _ = maximise(self.f, self.domain, seed=self.seed)

        # Take .value to return a float and test for exact equality
        self.assertEqual(x1.value, x2.value)

    def test_function_arg_errors(self):
        """A ValueError is raised if the provided function does not accept Inputs as
        args or does not return a real number."""

        # Does not accept Numpy arrays as args
        def f(x: np.ndarray) -> float:
            return np.sum(x**2)

        with self.assertRaisesRegex(
            ValueError,
            exact(
                f"Expected 'func' to be a callable that takes an argument of type {Input.__name__}."
            ),
        ):
            _ = maximise(f, self.domain)

        # Returns non-real objects
        def f(x: Input):
            return np.array([sum(x)])

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

    def test_failed_convergence_error(self):
        """A RuntimeError is raised if convergence failed in the maximisation."""

        def f(x):
            return np.random.random()

        with self.assertRaisesRegex(
            RuntimeError,
            "^Maximisation failed to converge: ",
        ):
            _ = maximise(f, self.domain, seed=self.seed)

class TestGenerateSeeds(unittest.TestCase):

    def setUp(self) -> None:
        self.seed = 1
        self.batch_size = 5

    def test_type_errors(self):
        """A TypeError is raised if the supplied seed is not an integer or None
        
        A TypeError is raised if the supplied batch_size is not an integer."""

        arg = "a"
        with self.assertRaisesRegex(
            TypeError, 
            exact(
                f"Expected 'seed' to be None or of type int, but received {type(arg)} instead."
            )
        ):
            _ = generate_seeds(arg, self.batch_size)

        with self.assertRaisesRegex(
            TypeError,
            exact(f"Expected 'batch_size' to be of type int, but received {type(arg)} instead."
            )
        ):
            _ = generate_seeds(self.seed, arg)

    def test_value_errors(self):
        """A ValueError is raised if the supplied seed is not a positive integer
        
        A ValueError is raised if the supplied batch_size is not a positive integer
        >=1 and < 1e9"""

        arg = -2
        with self.assertRaisesRegex(
            ValueError, 
            exact(
                f"Expected 'seed' to be None or >=0, but received {arg} instead."
            )
        ):
            _ = generate_seeds(arg, self.batch_size)

        with self.assertRaisesRegex(
            ValueError, 
            exact(
                f"Expected 'batch_size' to be >=1 and <1e9, but received {arg} instead."
            )
        ):
            _ = generate_seeds(self.seed, arg)

        arg2 = int(1e10)
        with self.assertRaisesRegex(
            ValueError, 
            exact(
                f"Expected 'batch_size' to be >=1 and <1e9, but received {arg2} instead."
            )
        ):
            _ = generate_seeds(self.seed, arg2)

    def test_length_seeds(self):
        """Ensures that the length of the returned seeds is the same as the batch_size passed"""

        seeds = generate_seeds(self.seed, self.batch_size)

        self.assertTrue(
            len(seeds) == self.batch_size
        )

    def test_return_type_int(self):
        """Ensures that all elements returned are integers within a tuple"""

        seeds = generate_seeds(self.seed, self.batch_size)

        self.assertIsInstance(seeds, tuple)

        for seed in seeds: 
            self.assertIsInstance(seed, int)

    def test_return_type_none(self):
        """Ensures that all elements returned are None within a tuple
        if None is passed as the seed""" 

        seeds = generate_seeds(None, self.batch_size)

        self.assertIsInstance(seeds, tuple)

        for seed in seeds:
            self.assertEqual(seed, None)

    def test_unique_seeds(self):
        """Ensures all returned seeds are unique"""

        # By poisson distribution this gives 99+% chance of having duplicates
        # but keeps the test at a sensible time
        batch_size = int(1e5)
        seeds = generate_seeds(self.seed, batch_size)

        self.assertTrue(
            len(set(seeds)) == batch_size
        )


if __name__ == "__main__":
    unittest.main()
