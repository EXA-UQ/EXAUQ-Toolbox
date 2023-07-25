import pathlib
import unittest

from exauq.core.modelling import Input
from exauq.core.simulators import Simulator


class TestSimulator(unittest.TestCase):
    def test_initialise_with_simulations_record_file(self):
        """Test that a simulator can be initialised with a handle to a file containing
        records of previous simulations."""

        _ = Simulator(r"a/b/c")  # Unix
        _ = Simulator(br"a/b/c")
        _ = Simulator(r"a\b\c")  # Windows
        _ = Simulator(br"a\b\c")
        _ = Simulator(pathlib.Path("a/b/c"))  # Platform independent
    
    def test_compute_output_unseen_input(self):
        """Test that None is returned as the value of a computation if a new input is
        supplied."""

        simulator = Simulator(r"a/b/c")
        self.assertIsNone(simulator.compute(Input(1)))


if __name__ == "__main__":
    unittest.main()
