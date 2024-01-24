import unittest

import numpy as np

from exauq.core.modelling import Input, SimulatorDomain
from exauq.utilities.optimisation import maximise
from tests.utilities.utilities import ExauqTestCase


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


if __name__ == "__main__":
    unittest.main()
