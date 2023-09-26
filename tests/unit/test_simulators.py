import csv
import os
import pathlib
import tempfile
import unittest.mock
from numbers import Real
from typing import Any, Optional, Type

from exauq.core.modelling import Input, SimulatorDomain
from exauq.core.simulators import SimulationsLog, SimulationsLogLookupError, Simulator
from exauq.core.types import FilePath
from tests.unit.fakes import DumbHardwareInterface, DumbJobManager
from tests.utilities.utilities import exact


def make_fake_simulator(
    simulations: tuple[tuple[Input, Real]],
    simulations_log: str = r"a/b/c.csv",
) -> Simulator:
    """Make a Simulator object whose previous simulations are pre-loaded to be the given
    simulations, if a path to a simulations log file is supplied (the default). If
    `simulations_log` is ``None`` then the simulator returned has no previous simulations.
    """

    with unittest.mock.patch(
        "exauq.core.simulators.SimulationsLog",
        new=make_fake_simulations_log_class(simulations),
    ), unittest.mock.patch("exauq.core.simulators.JobManager", new=DumbJobManager):
        return Simulator(
            SimulatorDomain([(-10, 10)]), DumbHardwareInterface(), simulations_log
        )


def make_fake_simulations_log_class(
    simulations: tuple[tuple[Input, Real]]
) -> Type[SimulationsLog]:
    """Make a class that fakes SimulationsLog by returning the prescribed
    simulations."""

    class FakeSimulationsLog(SimulationsLog):
        def __init__(self, file: FilePath, num_inputs: int, *args):
            super().__init__(file, num_inputs, *args)
            self.simulations = simulations

        def _initialise_log_file(self, file: FilePath) -> FilePath:
            """Return the path without creating a file there."""

            return file

        def get_simulations(self):
            """Return the pre-loaded simulations."""

            return self.simulations

        def add_new_record(self, x: Input):
            """Add the input to the list of simulations, with ``None`` as output. This
            represents a simulation being added as a pending job."""

            self.simulations += ((x, None),)

        def insert_result(self, job_id, result):
            pass

        def get_pending_jobs(self):
            """Return an empty list, indicating that there are no outstanding jobs."""

            return []

    return FakeSimulationsLog


def write_to_csv(
    data: dict[str, list], path: FilePath, write_header: Optional[bool] = True
) -> str:
    """Write data to a CSV file.

    `data` should be a dict with keys being the column headers and values being a list
    giving the column values. E.g.

    ``data = {'col1': [1, 2, 3], 'col2': ['dog', 'cat', 'fish']}``

    A value of ``None`` is converted to the empty string.
    """

    header = (
        sorted([col for col in data.keys() if col.startswith("Input_")])
        + ["Output"]
        + ["Job_ID"]
    )

    def to_str(x):
        output = str(x) if x is not None else ""
        return output

    with open(path, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        if write_header:
            writer.writeheader()
        writer.writerows(
            [
                {col: to_str(data[col][i]) for col in header}
                for i in range(len(data[header[0]]))
            ]
        )


def get_num_inputs(data: dict[str, Any]):
    return len(list(filter(lambda x: x.startswith("Input_"), data.keys())))


class TestSimulator(unittest.TestCase):
    def setUp(self) -> None:
        self.simulator_domain = SimulatorDomain([(-10, 10)])
        self.hardware_interface = DumbHardwareInterface()
        self.simulations = ((Input(1), 0),)
        self.empty_simulator = make_fake_simulator(tuple())
        self.simulator_with_sim = make_fake_simulator(self.simulations)

    def test_initialise_incorrect_types(self):
        """Test that a TypeError is raised if one of the arguments passed to the
        initialiser is of the incorrect type."""

        domain = 1
        with self.assertRaisesRegex(
            TypeError,
            exact(
                "Argument 'domain' must define a SimulatorDomain, but received object "
                f"of type {type(domain)} instead."
            ),
        ):
            Simulator(domain, self.hardware_interface)

        interface = 1
        with self.assertRaisesRegex(
            TypeError,
            exact(
                "Argument 'interface' must inherit from HardwareInterface, but received "
                f"object of type {type(interface)} instead."
            ),
        ):
            Simulator(self.simulator_domain, interface)

    def test_initialise_invalid_log_file_error(self):
        """Test that a TypeError is raised if an invalid path is supplied for the log
        file."""

        for path in [None, 0, 1]:
            with self.subTest(path=path):
                with self.assertRaisesRegex(
                    TypeError,
                    exact(
                        "Argument 'simulations_log' must define a file path, but received "
                        f"object of type {type(path)} instead."
                    ),
                ):
                    _ = Simulator(self.simulator_domain, self.hardware_interface, path)

    def test_initialise_with_simulations_record_file(self):
        """Test that a simulator can be initialised with a path to a file containing
        records of previous simulations."""

        with unittest.mock.patch(
            "exauq.core.simulators.SimulationsLog",
            new=make_fake_simulations_log_class(tuple()),
        ):
            # Unix
            _ = Simulator(self.simulator_domain, self.hardware_interface, r"a/b/c")
            _ = Simulator(self.simulator_domain, self.hardware_interface, rb"a/b/c")

            # Windows
            _ = Simulator(self.simulator_domain, self.hardware_interface, r"a\b\c")
            _ = Simulator(self.simulator_domain, self.hardware_interface, rb"a\b\c")

            # Platform independent
            _ = Simulator(
                self.simulator_domain, self.hardware_interface, pathlib.Path("a/b/c")
            )

    def test_initialise_default_log_file(self):
        """Test that a new log file with name 'simulations.csv' is created in the
        working directory as the default."""

        with unittest.mock.patch("exauq.core.simulators.SimulationsLog") as mock:
            _ = Simulator(self.simulator_domain, self.hardware_interface)
            mock.assert_called_once_with("simulations.csv", self.simulator_domain.dim)

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
        """Test that a TypeError is raised if an argument of type other than Input is
        supplied for computation."""

        for x in ["a", 1, (0, 0)]:
            with self.subTest(x=x):
                with self.assertRaisesRegex(
                    TypeError,
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

    def test_compute_for_computed_input_previous_sims_unchanged(self):
        """Test that, when compute is called on an input for which an output has
        been computed, the collection of previous simulations remains unchanged."""

        previous_sims = self.simulator_with_sim.previous_simulations
        _ = self.simulator_with_sim.compute(previous_sims[0][0])
        self.assertEqual(previous_sims, self.simulator_with_sim.previous_simulations)

    def test_compute_for_submitted_input_previous_sims_unchanged(self):
        """Test that, when compute is called on an input for which the computed result
        is pending, the collection of previous simulations remains unchanged."""

        simulations = ((Input(1), None),)
        simulator = make_fake_simulator(simulations)
        _ = simulator.compute(simulations[0][0])
        self.assertEqual(simulations, simulator.previous_simulations)

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
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_dir = pathlib.Path(self.tmp.name).as_posix()
        self.simulations_file = pathlib.Path(self.tmp_dir, "simulations.csv")

        self.log_data1 = {"Input_1": [], "Output": [], "Job_ID": []}
        self.log_data2 = {
            "Input_1": [1, 2],
            "Output": [10, None],
            "Job_ID": [1, 2],
        }

    def tearDown(self) -> None:
        super().tearDown()
        self.tmp.cleanup()

    def make_simulations_log(self, data: dict[str, list]):
        """Create a simulations log file pre-populated with the given data."""

        write_to_csv(data, self.simulations_file)
        return SimulationsLog(self.simulations_file, get_num_inputs(data))

    def test_initialise_invalid_log_file_error(self):
        """Test that a TypeError is raised if a simulator log is initialised without a
        valid path to a log file."""

        for path in [None, 0, 1]:
            with self.subTest(path=path):
                with self.assertRaisesRegex(
                    TypeError,
                    exact(
                        "Argument 'file' must define a file path, but received object of "
                        f"type {type(path)} instead."
                    ),
                ):
                    _ = SimulationsLog(path, num_inputs=3)

    def test_initialise_with_simulations_record_file(self):
        """Test that a simulator log can be initialised with a handle to the log file."""

        # Unix
        path = str(pathlib.PurePosixPath(self.tmp_dir, "simulations.csv"))
        _ = SimulationsLog(path, num_inputs=3)
        _ = SimulationsLog(path.encode(), num_inputs=3)

        # Windows
        path = str(pathlib.PureWindowsPath(self.tmp_dir, "simulations.csv"))
        _ = SimulationsLog(path, num_inputs=3)
        _ = SimulationsLog(path.encode(), num_inputs=3)

        # Platform independent
        _ = SimulationsLog(pathlib.Path(self.tmp_dir, "simulations.csv"), num_inputs=3)

    def test_initialise_new_log_file_created(self):
        """Test that a new simulator log file at a given path is created upon object
        initialisation."""

        _ = SimulationsLog(self.simulations_file, num_inputs=3)
        self.assertTrue(self.simulations_file.exists())

    def test_initialise_new_log_file_not_opened_if_exists(self):
        """Test that an existing simulator log file is not opened for writing upon
        initialisation."""

        write_to_csv(self.log_data1, self.simulations_file)
        self.simulations_file.touch(mode=0o400)  # read-only
        try:
            _ = SimulationsLog(self.simulations_file, get_num_inputs(self.log_data1))
        except PermissionError:
            self.fail("Tried writing to pre-existing log file, should not have done.")
        finally:
            os.chmod(self.simulations_file, 0o600)  # read/write mode, to allow deletion
            os.remove(self.simulations_file)

    def test_get_simulations_no_simulations_in_file(self):
        """Test that an empty tuple is returned if the simulations log file does not
        contain any simulations."""

        log = self.make_simulations_log(self.log_data1)
        self.assertEqual(tuple(), log.get_simulations())

    def test_get_simulations_one_dim_input(self):
        """Test that simulation data with a 1-dimensional input is parsed correctly."""

        data = {"Input_1": [1], "Output": [2], "Job_ID": [1]}
        log = self.make_simulations_log(data)
        expected = ((Input(1), 2),)
        self.assertEqual(expected, log.get_simulations())

    def test_get_simulations_returns_simulations_from_file(self):
        """Test that a record of all simulations (pending or otherwise) recorded
        in the log file are returned."""

        data = {
            "Input_1": [1, 3],
            "Input_2": [2, 4],
            "Output": [10, None],
            "Job_ID": [1, 2],
        }
        log = self.make_simulations_log(data)
        expected = ((Input(1, 2), 10), (Input(3, 4), None))
        self.assertEqual(expected, log.get_simulations())

    def test_get_simulations_unusual_column_order(self):
        """Test that a log file is parsed correctly irrespective of the order of input
        and output columns."""

        self.simulations_file.write_text("Input_2,Job_ID,Output,Input_1\n1,0,2,10\n")
        log = SimulationsLog(self.simulations_file, num_inputs=2)
        expected = ((Input(10, 1), 2),)
        self.assertEqual(expected, log.get_simulations())

    def test_add_new_record_single_input(self):
        """Test that, when a record for a given input is added, the corresponding
        simulation shows up in the list of previous simulations."""

        x = Input(1)
        log = SimulationsLog(self.simulations_file, num_inputs=len(x))
        log.add_new_record(x)
        self.assertEqual(((x, None),), log.get_simulations())

    def test_add_new_record_multiple_records_same_input(self):
        """Test that, when multiple records for the same input are added, one simulation
        for each record shows up in the list of previous simulations."""

        x = Input(1)
        log = SimulationsLog(self.simulations_file, num_inputs=len(x))
        log.add_new_record(x)
        log.add_new_record(x)
        self.assertEqual(((x, None), (x, None)), log.get_simulations())

    def test_add_new_record_multiple_records_different_inputs(self):
        """Test that, when records for different inputs are added, one simulation
        for each record shows up in the list of previous simulations."""

        x1 = Input(1)
        x2 = Input(2)
        log = SimulationsLog(self.simulations_file, num_inputs=len(x1))
        log.add_new_record(x1)
        log.add_new_record(x2)
        self.assertEqual(((x1, None), (x2, None)), log.get_simulations())

    def test_get_unsubmitted_inputs_no_inputs(self):
        """Test that an empty tuple is returned if no inputs have been submitted to the
        log file."""

        log = SimulationsLog(self.simulations_file, num_inputs=1)
        self.assertEqual(tuple(), log.get_unsubmitted_inputs())

    def test_get_unsubmitted_inputs_unsubmitted_input(self):
        """Test that, when an input is submitted with no job ID, it features as an
        unsubmitted input."""

        x = Input(1)
        log = SimulationsLog(self.simulations_file, num_inputs=len(x))
        log.add_new_record(x)
        self.assertEqual((x,), log.get_unsubmitted_inputs())

    def test_get_unsubmitted_inputs_submitted_input(self):
        """Test that, when an input is submitted with a job ID, it does not feature as
        an unsubmitted input."""

        x = Input(1)
        job_id = "0"
        log = SimulationsLog(self.simulations_file, num_inputs=len(x))
        log.add_new_record(x, job_id)
        self.assertTrue(x not in log.get_unsubmitted_inputs())

    def test_get_unsubmitted_inputs_submitted_unsubmitted_mix(self):
        """Test that, when several inputs are submitted, only those that weren't
        submitted with a job ID are returned as unsubmitted inputs."""

        x1 = Input(1)
        x2 = Input(2)
        x3 = Input(3)
        x4 = Input(4)
        log = SimulationsLog(self.simulations_file, num_inputs=len(x1))
        log.add_new_record(x1, job_id="0")
        log.add_new_record(x2)
        log.add_new_record(
            x3,
        )
        log.add_new_record(x4, job_id="1")
        self.assertEqual((x2, x3), log.get_unsubmitted_inputs())

    def test_get_records_single_job_id(self):
        """Test that the record with a specified job ID can be successfully retrieved."""

        log = self.make_simulations_log(self.log_data2)
        job_id = "2"
        expected = ({"Input_1": "2", "Output": "", "Job_ID": job_id},)
        self.assertEqual(expected, log.get_records({job_id}))

    def test_get_records_multiple_job_ids(self):
        """Test that records with a specified job IDs can be successfully retrieved, in
        the case where multiple records are requested."""

        log = self.make_simulations_log(self.log_data2)
        job_ids = {"1", "2"}
        expected = (
            {"Input_1": "1", "Output": "10", "Job_ID": "1"},
            {"Input_1": "2", "Output": "", "Job_ID": "2"},
        )
        self.assertEqual(expected, log.get_records(job_ids))

    def test_get_records_all_records(self):
        """Test that all records are retrieved when no job IDs are specified."""

        log = self.make_simulations_log(self.log_data2)
        expected = (
            {"Input_1": "1", "Output": "10", "Job_ID": "1"},
            {"Input_1": "2", "Output": "", "Job_ID": "2"},
        )
        self.assertEqual(expected, log.get_records())

    def test_create_record_one_dim_input(self):
        """Test that a record can be added to the simulations log file, in the
        case where the input space is one-dimensional."""

        record = {"Input_1": "1.1", "Output": "2", "Job_ID": "1"}
        log = SimulationsLog(self.simulations_file, get_num_inputs(record))
        log.create_record(record)
        self.assertEqual((record,), log.get_records(record["Job_ID"]))

    def test_create_record_multi_dim_input(self):
        """Test that a record can be added to the simulations log file, in the
        case where the input space is multi-dimensional."""

        record = {"Input_1": "1.1", "Input_2": "2.2", "Output": "2", "Job_ID": "1"}
        log = SimulationsLog(self.simulations_file, get_num_inputs(record))
        log.create_record(record)
        self.assertEqual((record,), log.get_records(record["Job_ID"]))

    def test_create_record_appends(self):
        """Test that multiple applications of creating records appends the records to
        the simulations log file."""

        record1 = {"Input_1": "1.1", "Output": "2", "Job_ID": "1"}
        record2 = {"Input_1": "2.2", "Output": "2", "Job_ID": "2"}
        log = SimulationsLog(self.simulations_file, get_num_inputs(record1))
        log.create_record(record1)
        log.create_record(record2)
        self.assertEqual((record1, record2), log.get_records({"1", "2"}))

    def test_create_record_missing_field_error(self):
        """Test that a ValueError is raised if one of the log file fields is missing
        from it."""

        record = {"Input_2": "1.1", "Output": "2", "Job_ID": "1"}
        log = SimulationsLog(self.simulations_file, get_num_inputs(record))
        with self.assertRaisesRegex(
            ValueError,
            exact(
                "The record does not contain entries for the required fields: Input_1."
            ),
        ):
            log.create_record(record)

    def test_create_record_extra_field_error(self):
        """Test that a ValueError is raised if the record contains a field that is not
        in the simulations log file header."""

        record = {"Input_1": "1", "Input_2": "1.1", "Output": "2", "Job_ID": "1"}
        log = SimulationsLog(self.simulations_file, get_num_inputs(record) - 1)
        with self.assertRaisesRegex(
            ValueError,
            exact("The record contains fields not in the simulations log file: Input_2."),
        ):
            log.create_record(record)

    def test_insert_result_output_added(self):
        """Test that a simulator output is added alongside a simulator input and
        can be retrieved from the previous simulations."""

        x = Input(1)
        y = 10
        job_id = "0"
        log = SimulationsLog(self.simulations_file, num_inputs=len(x))
        log.add_new_record(x, job_id)
        log.insert_result(job_id, y)
        self.assertEqual(((x, y),), log.get_simulations())

    def test_insert_result_missing_job_id_error(self):
        """Test that a SimulationsLogLookupError is raised if one attempts to add an
        output with a job ID that doesn't exist in the simulations log file."""

        x = Input(1)
        log = SimulationsLog(self.simulations_file, num_inputs=len(x))
        log.add_new_record(x, "0")
        job_id = "1"
        with self.assertRaisesRegex(
            SimulationsLogLookupError,
            exact(
                f"Could not add output to simulation with job ID = {job_id}: "
                "no such simulation exists."
            ),
        ):
            log.insert_result(job_id, 10)

    def test_insert_result_multiple_job_id_error(self):
        """Test that a SimulationsLogLookupError is raised if there are multiple
        records with the same job ID when trying to insert a simulator output."""

        x = Input(1)
        log = SimulationsLog(self.simulations_file, num_inputs=len(x))
        job_id = "0"
        log.add_new_record(x, job_id)
        log.add_new_record(x, job_id)
        with self.assertRaisesRegex(
            SimulationsLogLookupError,
            exact(
                f"Could not add output to simulation with job ID = {job_id}: "
                "multiple records with this ID found."
            ),
        ):
            log.insert_result(job_id, 10)

    def test_get_pending_jobs_empty_log_file(self):
        """Test that an empty tuple is returned if there are no records in the
        simulations log file."""

        log = SimulationsLog(self.simulations_file, num_inputs=1)
        self.assertEqual(tuple(), log.get_pending_jobs())

    def test_get_pending_jobs_empty_when_all_completed(self):
        """Test that an empty tuple is returned if all jobs in the simulations
        log file have a simulator output."""

        data = {"Input_1": [1, 2], "Output": [10.1, 20.2], "Job_ID": ["1", "2"]}
        log = self.make_simulations_log(data)
        self.assertEqual(tuple(), log.get_pending_jobs())

    def test_get_pending_jobs_selects_correct_jobs(self):
        """Test that the jobs selected are those that have a Job ID but not an
        output."""

        data = {
            "Input_1": [1, 2, 3, 4],
            "Output": [10.1, None, None, None],
            "Job_ID": ["1", "2", "3", None],
        }
        log = self.make_simulations_log(data)
        self.assertEqual(("2", "3"), log.get_pending_jobs())


if __name__ == "__main__":
    unittest.main()
