import unittest

import numpy as np

from exauq.core.modelling import SimulatorDomain
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


if __name__ == "__main__":
    unittest.main()
