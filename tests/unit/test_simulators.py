import os
import pathlib
import tempfile
import unittest.mock
from numbers import Real
from typing import Type

from exauq.core.modelling import Input
from exauq.core.simulators import SimulationsLog, Simulator
from exauq.core.types import FilePath
from tests.utilities.utilities import exact


def make_fake_simulations_log_class(
    simulations: tuple[tuple[Input, Real]]
) -> Type[SimulationsLog]:
    """Make a class that fakes SimulationsLog by returning the prescribed
    simulations."""

    class FakeSimulationsLog(SimulationsLog):
        def __init__(self, file: FilePath, *args):
            super().__init__(file, *args)
            self.simulations = simulations

        @staticmethod
        def _initialise_log_file(file: FilePath) -> FilePath:
            """Return the path without creating a file there."""

            return file

        def get_simulations(self):
            """Return the pre-loaded simulations."""

            return self.simulations

    return FakeSimulationsLog


def make_fake_simulator(
    simulations: tuple[tuple[Input, Real]],
    simulations_log: str = r"a/b/c.log",
) -> Simulator:
    """Make a Simulator object whose previous simulations are pre-loaded to be the given
    simulations, if a path to a simulations log file is supplied (the default). If
    `simulations_log` is ``None`` then the simulator returned has no previous simulations.
    """

    with unittest.mock.patch(
        "exauq.core.simulators.SimulationsLog",
        new=make_fake_simulations_log_class(simulations),
    ):
        return Simulator(simulations_log)


class TestSimulator(unittest.TestCase):
    def setUp(self) -> None:
        self.simulations = ((Input(1), 0),)
        self.empty_simulator = make_fake_simulator(tuple())
        self.simulator_with_sim = make_fake_simulator(self.simulations)

    def test_initialise_invalid_log_file_error(self):
        """Test that a ValueError is raised if an invalid path is supplied for the log
        file."""

        for path in [None, 0, 1]:
            with self.subTest(path=path):
                with self.assertRaisesRegex(
                    ValueError,
                    exact(
                        f"Argument 'simulations_log' must define a file path, got {path} "
                        "instead."
                    ),
                ):
                    _ = Simulator(path)

    def test_initialise_with_simulations_record_file(self):
        """Test that a simulator can be initialised with a path to a file containing
        records of previous simulations."""

        with unittest.mock.patch(
            "exauq.core.simulators.SimulationsLog",
            new=make_fake_simulations_log_class(tuple()),
        ):
            _ = Simulator(r"a/b/c")  # Unix
            _ = Simulator(rb"a/b/c")
            _ = Simulator(r"a\b\c")  # Windows
            _ = Simulator(rb"a\b\c")
            _ = Simulator(pathlib.Path("a/b/c"))  # Platform independent

    def test_initialise_default_log_file(self):
        """Test that a new log file with name 'simulations.csv' is created in the
        working directory as the default."""

        with unittest.mock.patch("exauq.core.simulators.SimulationsLog") as mock:
            _ = Simulator()
            mock.assert_called_once_with("simulations.csv")

    def test_previous_simulations_no_simulations_run(self):
        """Test that an empty tuple is returned if there are no simulations in the
        log file."""

        self.assertEqual(tuple(), self.empty_simulator.previous_simulations)

    def test_previous_simulations_immutable(self):
        """Test that the previous_simulations property is read-only."""

        with self.assertRaises(AttributeError):
            self.empty_simulator.previous_simulations = tuple()

    def test_previous_simulations_from_log_file(self):
        """Test that the previously run simulations are returned from the log file
        when this is supplied."""

        self.assertEqual(self.simulations, self.simulator_with_sim.previous_simulations)

    def test_compute_non_input_error(self):
        """Test that a ValueError is raised if an argument of type other than Input is
        supplied for computation."""

        for x in ["a", 1, (0, 0)]:
            with self.subTest(x=x):
                with self.assertRaisesRegex(
                    ValueError,
                    exact(f"Argument 'x' must be of type Input, but received {type(x)}."),
                ):
                    self.empty_simulator.compute(x)

    def test_compute_output_unseen_input(self):
        """Test that None is returned as the value of a computation if a new input is
        supplied."""

        self.assertIsNone(self.empty_simulator.compute(Input(2)))

    def test_compute_new_input_features_in_previous_simulations(self):
        """Test that, when an unseen input is submitted for computation, the
        input features in the previous simulations."""

        x = Input(2)
        self.empty_simulator.compute(x)
        self.assertEqual(x, self.empty_simulator.previous_simulations[-1][0])

    def test_compute_new_input_features_in_previous_simulations_multiple(self):
        """Test that, when multiple unseen inputs are submitted for computation
        sequentially, these inputs feature in the previous simulations."""

        x1 = Input(2)
        x2 = Input(3)
        self.empty_simulator.compute(x1)
        self.empty_simulator.compute(x2)
        self.assertEqual(x2, self.empty_simulator.previous_simulations[-1][0])
        self.assertEqual(x1, self.empty_simulator.previous_simulations[-2][0])

    def test_compute_returns_output_for_computed_input(self):
        """Test that, when compute is called on an input for which an output has
        been computed, this output is returned."""

        x, y = self.simulator_with_sim.previous_simulations[0]
        self.assertEqual(y, self.simulator_with_sim.compute(x))

    def test_compute_returns_output_for_computed_input_multiple_simulations(self):
        """Test that, when compute is called on an input for which an output has
        been computed, this output is returned, in the case where there are multiple
        previous simulations."""

        simulations = ((Input(1), 0), (Input(2), 10))
        simulator = make_fake_simulator(simulations)
        for x, y in simulator.previous_simulations:
            with self.subTest(x=x, y=y):
                self.assertEqual(y, simulator.compute(x))


class TestSimulationsLog(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.simulations_file = "foo.csv"
        with unittest.mock.patch("builtins.open", unittest.mock.mock_open()):
            self.log = SimulationsLog(self.simulations_file)

    def assert_file_opened(self, mock_open, file_path, mode="r"):
        """Check that a mocked ``open()`` is called once on the specified file path in
        the given mode ("read" by default")."""

        mock_open.assert_called_once()
        self.assertEqual((file_path,), mock_open.call_args.args)
        self.assertTrue(("mode", mode) in mock_open.call_args.kwargs.items())

    def test_initialise_invalid_log_file_error(self):
        """Test that a ValueError is raised if a simulator log is initialised without a
        valid path to a log file."""

        for path in [None, 0, 1]:
            with self.subTest(path=path):
                with self.assertRaisesRegex(
                    ValueError,
                    exact(
                        f"Argument 'file' must define a file path, got {path} instead."
                    ),
                ):
                    _ = SimulationsLog(path)

    def test_initialise_with_simulations_record_file(self):
        """Test that a simulator log can be initialised with a handle to the log file."""

        with unittest.mock.patch(
            "builtins.open", unittest.mock.mock_open(read_data="Input_1,Output\n")
        ):
            _ = SimulationsLog(r"a/b/c.log")  # Unix
            _ = SimulationsLog(rb"a/b/c.log")
            _ = SimulationsLog(r"a\b\c.log")  # Windows
            _ = SimulationsLog(rb"a\b\c.log")
            _ = SimulationsLog(pathlib.Path("a/b/c.log"))  # Platform independent

    def test_initialise_new_log_file_created(self):
        """Test that a new simulator log file at a given path is created upon object
        initialisation."""

        file_path = pathlib.Path("a/b/c.log")
        with unittest.mock.patch("builtins.open", unittest.mock.mock_open()) as mock:
            _ = SimulationsLog(file_path)
            self.assert_file_opened(mock, file_path, mode="w")

    def test_initialise_new_log_file_not_opened_if_exists(self):
        """Test that an existing simulator log file is not opened for writing upon
        initialisation."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = pathlib.Path(tmp_dir, "log.csv")
            path.touch(mode=0o400)  # read-only
            try:
                _ = SimulationsLog(path)
            except PermissionError:
                self.fail("Tried writing to pre-existing log file, should not have done.")
            finally:
                os.chmod(path, 0o600)  # read/write mode, to allow deletion
                os.remove(path)

    def test_get_simulations_no_simulations_in_file(self):
        """Test that an empty tuple is returned if the simulations log file does not
        contain any simulations."""

        with unittest.mock.patch(
            "builtins.open", unittest.mock.mock_open(read_data="Input_1,Output\n")
        ) as mock:
            self.assertEqual(tuple(), self.log.get_simulations())
            self.assert_file_opened(mock, self.simulations_file)

    def test_get_simulations_one_dim_input(self):
        """Test that simulation data with a 1-dimensional input is parsed correctly."""

        with unittest.mock.patch(
            "builtins.open",
            unittest.mock.mock_open(read_data="Input_1,Output\n1,2\n"),
        ) as mock:
            expected = ((Input(1), 2),)
            self.assertEqual(expected, self.log.get_simulations())
            self.assert_file_opened(mock, self.simulations_file)

    def test_get_simulations_returns_simulations_from_file(self):
        """Test that a record of all simulations (pending or otherwise) recorded
        in the log file are returned."""

        with unittest.mock.patch(
            "builtins.open",
            unittest.mock.mock_open(read_data="Input_1,Input_2,Output\n1,2,10\n3,4,\n"),
        ) as mock:
            expected = ((Input(1, 2), 10), (Input(3, 4), None))
            self.assertEqual(expected, self.log.get_simulations())
            self.assert_file_opened(mock, self.simulations_file)

    def test_get_simulations_unusual_column_order(self):
        """Test that a log file is parsed correctly irrespective of the order of input
        and output columns."""

        with unittest.mock.patch(
            "builtins.open",
            unittest.mock.mock_open(read_data="Input_2,Output,Input_1\n1,2,10\n"),
        ) as mock:
            expected = ((Input(10, 1), 2),)
            self.assertEqual(expected, self.log.get_simulations())
            self.assert_file_opened(mock, self.simulations_file)


if __name__ == "__main__":
    unittest.main()
