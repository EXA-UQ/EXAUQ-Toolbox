import pathlib
import unittest.mock
from numbers import Real

from exauq.core.modelling import Input
from exauq.core.simulators import SimulationsLog, Simulator


def make_fake_simulations_log_class(simulations: tuple[tuple[Input, Real]]):
    """Make a class that fakes SimulationsLog by returning the prescribed
    simulations."""

    class FakeSimulationsLog(SimulationsLog):
        def __init__(self, *args):
            super().__init__(*args)
            self.simulations = simulations if self._log_file else ()

        def get_simulations(self):
            return self.simulations

    return FakeSimulationsLog


class TestSimulator(unittest.TestCase):
    def setUp(self) -> None:
        self.simulations = ((Input(1), 0),)
        self.FakeSimulationsLog = make_fake_simulations_log_class(self.simulations)

        with unittest.mock.patch(
            "exauq.core.simulators.SimulationsLog", new=self.FakeSimulationsLog
        ):
            self.simulator = Simulator()

    def test_initialise_with_simulations_record_file(self):
        """Test that a simulator can be initialised with a path to a file containing
        records of previous simulations."""

        with unittest.mock.patch(
            "exauq.core.simulators.SimulationsLog", new=self.FakeSimulationsLog
        ):
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

    def test_previous_simulations_from_log_file(self):
        """Test that the previously run simulations are returned from the log file
        when this is supplied."""

        with unittest.mock.patch(
            "exauq.core.simulators.SimulationsLog", new=self.FakeSimulationsLog
        ):
            simulator = Simulator(r"a/b/c.log")
            self.assertEqual(self.simulations, simulator.previous_simulations)

    def test_compute_output_unseen_input(self):
        """Test that None is returned as the value of a computation if a new input is
        supplied."""

        self.assertIsNone(self.simulator.compute(Input(2)))

    def test_compute_new_input_features_in_previous_simulations(self):
        """Test that, when an unseen input is submitted for computation, the
        input features in the previous simulations."""

        x = Input(2)
        self.simulator.compute(x)
        self.assertEqual(x, self.simulator.previous_simulations[-1][0])

    def test_compute_new_input_features_in_previous_simulations_multiple(self):
        """Test that, when multiple unseen inputs are submitted for computation
        sequentially, these inputs feature in the previous simulations."""

        x1 = Input(2)
        x2 = Input(3)
        self.simulator.compute(x1)
        self.simulator.compute(x2)
        self.assertEqual(x2, self.simulator.previous_simulations[-1][0])
        self.assertEqual(x1, self.simulator.previous_simulations[-2][0])


class TestSimulationsLog(unittest.TestCase):
    def setUp(self) -> None:
        self.simulations_file = "foo.csv"
        self.log = SimulationsLog(self.simulations_file)

        # For testing log file row parsing
        self.x = Input(1, 2, 3)
        self.y = 100

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
            "builtins.open", unittest.mock.mock_open(read_data="Input_1,Output\n")
        ) as mock:
            self.assertEqual(tuple(), self.log.get_simulations())
            self.assert_file_read(mock, self.simulations_file)

    def test_get_simulations_one_dim_input(self):
        """Test that simulation data with a 1-dimensional input is parsed correctly."""

        with unittest.mock.patch(
            "builtins.open",
            unittest.mock.mock_open(read_data="Input_1,Output\n1,2\n"),
        ) as mock:
            expected = ((Input(1), 2),)
            self.assertEqual(expected, self.log.get_simulations())
            self.assert_file_read(mock, self.simulations_file)

    def test_get_simulations_returns_simulations_from_file(self):
        """Test that a record of all simulations (pending or otherwise) recorded
        in the log file are returned."""

        with unittest.mock.patch(
            "builtins.open",
            unittest.mock.mock_open(read_data="Input_1,Input_2,Output\n1,2,10\n3,4,\n"),
        ) as mock:
            expected = ((Input(1, 2), 10), (Input(3, 4), None))
            self.assertEqual(expected, self.log.get_simulations())
            self.assert_file_read(mock, self.simulations_file)

    def test_get_simulations_unusual_column_order(self):
        """Test that a log file is parsed correctly irrespective of the order of input
        and output columns."""

        with unittest.mock.patch(
            "builtins.open",
            unittest.mock.mock_open(read_data="Input_2,Output,Input_1\n1,2,10\n"),
        ) as mock:
            expected = ((Input(10, 1), 2),)
            self.assertEqual(expected, self.log.get_simulations())
            self.assert_file_read(mock, self.simulations_file)


if __name__ == "__main__":
    unittest.main()
