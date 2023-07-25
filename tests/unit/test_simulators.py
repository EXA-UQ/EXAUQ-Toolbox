import pathlib
import unittest

from exauq.core.modelling import Input
from exauq.core.simulators import Simulator


class TestSimulator(unittest.TestCase):
    def setUp(self) -> None:
        self.simulator = Simulator()

    def test_initialise_with_simulations_record_file(self):
        """Test that a simulator can be initialised with a handle to a file containing
        records of previous simulations."""

        _ = Simulator(r"a/b/c")  # Unix
        _ = Simulator(br"a/b/c")
        _ = Simulator(r"a\b\c")  # Windows
        _ = Simulator(br"a\b\c")
        _ = Simulator(pathlib.Path("a/b/c"))  # Platform independent

    def test_previous_simulations_no_simulations_run(self):
        """Test that an empty tuple is returned if there are no simulations in the
        log file."""

        self.assertEqual(tuple(), self.simulator.previous_simulations)

    def test_previous_simulations_immutable(self):
        """Test that the previous_simulations property is read-only."""

        with self.assertRaises(AttributeError):
            self.simulator.previous_simulations = tuple()

    def test_compute_output_unseen_input(self):
        """Test that None is returned as the value of a computation if a new input is
        supplied."""

        self.assertIsNone(self.simulator.compute(Input(1)))


if __name__ == "__main__":
    unittest.main()
