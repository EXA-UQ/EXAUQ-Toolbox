import unittest

from exauq.core.modelling import Input
from exauq.core.simulators import Simulator


class TestSimulator(unittest.TestCase):
    def test_compute_output_new_input_exist(self):
        """Test that None is returned as the value of a computation if a new input is
        supplied."""

        simulator = Simulator()
        self.assertIsNone(simulator.compute(Input(1)))


if __name__ == "__main__":
    unittest.main()
