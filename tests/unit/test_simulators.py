import pathlib
import unittest.mock

from exauq.core.modelling import Input
from exauq.core.simulators import SimulationsLog, Simulator


class TestSimulator(unittest.TestCase):
    def setUp(self) -> None:
        self.simulator = Simulator()

    def test_initialise_with_simulations_record_file(self):
        """Test that a simulator can be initialised with a handle to a file containing
        records of previous simulations."""

        _ = Simulator(r"a/b/c")  # Unix
        _ = Simulator(rb"a/b/c")
        _ = Simulator(r"a\b\c")  # Windows
        _ = Simulator(rb"a\b\c")
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


class TestSimulatorLog(unittest.TestCase):
    def setUp(self) -> None:
        self.simulations_file = "foo.csv"

    def assert_file_read(self, mock_open, file_path):
        """Check that a mocked ``open()`` is called once in read mode on the specified
        path."""

        mock_open.assert_called_once()
        self.assertEqual((file_path,), mock_open.call_args.args)
        self.assertTrue(("mode", "r") in mock_open.call_args.kwargs.items())

    def test_initialise_with_simulations_record_file(self):
        """Test that a simulator log can be initialised with a handle to the log file."""

        _ = SimulationsLog(r"a/b/c")  # Unix
        _ = SimulationsLog(rb"a/b/c")
        _ = SimulationsLog(r"a\b\c")  # Windows
        _ = SimulationsLog(rb"a\b\c")
        _ = SimulationsLog(pathlib.Path("a/b/c"))  # Platform independent

    def test_get_simulations_no_simulations_in_file(self):
        """Test that an empty tuple is returned if the simulations log file does not
        contain any simulations."""

        with unittest.mock.patch(
            "builtins.open", unittest.mock.mock_open(read_data="Input,Output\n")
        ) as mock:
            log = SimulationsLog(self.simulations_file)
            self.assertEqual(tuple(), log.get_simulations())
            self.assert_file_read(mock, self.simulations_file)

    def test_get_simulations_returns_simulations_from_file(self):
        """Test that a record of all simulations (pending or otherwise) recorded
        in the log file are returned."""

        with unittest.mock.patch(
            "builtins.open",
            unittest.mock.mock_open(read_data="Input,Output\n1,10\n2,\n"),
        ) as mock:
            log = SimulationsLog(self.simulations_file)
            expected = ((Input(1), 10), (Input(2), None))
            self.assertEqual(expected, log.get_simulations())
            self.assert_file_read(mock, self.simulations_file)


if __name__ == "__main__":
    unittest.main()
