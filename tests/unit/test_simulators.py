import os
import pathlib
import sys
import tempfile
import unittest.mock
from numbers import Real
from typing import Type

from exauq.core.modelling import Input, SimulatorDomain
from exauq.core.types import FilePath
from exauq.sim_management.hardware import JobStatus
from exauq.sim_management.jobs import JobId
from exauq.sim_management.simulators import (
    SimulationsLog,
    SimulationsLogLookupError,
    Simulator,
)
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
        "exauq.sim_management.simulators.SimulationsLog",
        new=make_fake_simulations_log_class(simulations),
    ), unittest.mock.patch(
        "exauq.sim_management.simulators.JobManager", new=DumbJobManager
    ):
        return Simulator(
            SimulatorDomain([(-10, 10)]), DumbHardwareInterface(), simulations_log
        )


def make_fake_simulations_log_class(
    simulations: tuple[tuple[Input, Real]]
) -> Type[SimulationsLog]:
    """Make a class that fakes SimulationsLog by returning the prescribed
    simulations."""

    class FakeSimulationsLog(SimulationsLog):
        def __init__(self, file: FilePath, input_dim: int, *args):
            super().__init__(file, input_dim, *args)
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
            "exauq.sim_management.simulators.SimulationsLog",
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

        with unittest.mock.patch(
            "exauq.sim_management.simulators.SimulationsLog"
        ) as mock:
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
                    _ = SimulationsLog(path, input_dim=3)

    @unittest.skipUnless(sys.platform.startswith("win"), "requires Windows")
    def test_initialise_new_log_file_created_windows(self):
        """Test that a new simulator log file at a given path is created upon object
        initialisation, on a Windows system."""

        _ = SimulationsLog(
            str(pathlib.PureWindowsPath(self.simulations_file)), input_dim=3
        )
        self.assertTrue(self.simulations_file.exists())

    @unittest.skipIf(sys.platform.startswith("win"), "requires POSIX-based system")
    def test_initialise_new_log_file_created(self):
        """Test that a new simulator log file at a given path is created upon object
        initialisation, on a POSIX-based system."""

        _ = SimulationsLog(str(pathlib.PurePosixPath(self.simulations_file)), input_dim=3)
        self.assertTrue(self.simulations_file.exists())

    def test_initialise_new_log_file_not_opened_if_exists(self):
        """Test that an existing simulator log file is not opened for writing upon
        initialisation."""

        self.simulations_file.write_text("Input_1,Output,Job_ID,Job_Status\n")
        self.simulations_file.touch(mode=0o400)  # read-only
        try:
            _ = SimulationsLog(self.simulations_file, input_dim=1)
        except PermissionError:
            self.fail("Tried writing to pre-existing log file, should not have done.")
        finally:
            os.chmod(self.simulations_file, 0o600)  # read/write mode, to allow deletion
            os.remove(self.simulations_file)

    def test_initialise_input_dim_type_error(self):
        """Test that a TypeError is raised if the input dimension supplied is not
        an integer."""

        for dim in (1.1, "1"):
            with self.subTest(dim=dim), self.assertRaisesRegex(
                TypeError,
                exact(
                    "Expected 'input_dim' to be of type integer, "
                    f"but received {type(dim)} instead."
                ),
            ):
                _ = SimulationsLog(self.simulations_file, dim)

    def test_initialise_bad_input_dim_error(self):
        """Test that a ValueError is raised if a non-positive integer is supplied
        as the input dimension."""

        for dim in (-1, 0):
            with self.subTest(dim=dim), self.assertRaisesRegex(
                ValueError,
                exact(
                    "Expected 'input_dim' to be a positive integer, "
                    f"but received {dim} instead."
                ),
            ):
                _ = SimulationsLog(self.simulations_file, dim)

    def test_get_simulations_no_simulations_in_file(self):
        """Test that an empty tuple is returned if the simulations log file does not
        contain any simulations."""

        log = SimulationsLog(self.simulations_file, input_dim=1)
        self.assertEqual(tuple(), log.get_simulations())

    def test_get_simulations_returns_simulations_from_file(self):
        """Test that a record of all simulations recorded in the log file are
        returned."""

        log = SimulationsLog(self.simulations_file, input_dim=2)

        x1 = Input(1, 1)
        log.add_new_record(x1, job_id="1")
        log.insert_result(job_id="1", result=10)

        x2 = Input(2, 2)
        log.add_new_record(x2, job_id="2")

        x3 = Input(3, 3)
        log.add_new_record(x3, job_id="3")

        expected = ((x1, 10), (x2, None), (x3, None))
        self.assertEqual(expected, log.get_simulations())

    def test_get_simulations_unusual_column_order(self):
        """Test that a log file is parsed correctly irrespective of the order of input
        and output columns."""

        self.simulations_file.write_text(
            "Input_2,Job_Status,Job_ID,Output,Input_1\n1,Completed,0,2,10\n"
        )
        log = SimulationsLog(self.simulations_file, input_dim=2)
        expected = ((Input(10, 1), 2),)
        self.assertEqual(expected, log.get_simulations())

    def test_add_new_record_input_wrong_dim_error(self):
        """Test that a ValueError is raised if the supplied input has a different number
        of coordinates to that expected of simulator inputs in the log file."""

        expected_dim = 2
        log = SimulationsLog(self.simulations_file, input_dim=expected_dim)
        inputs = (Input(1), Input(1, 1, 1))
        for x in inputs:
            with self.subTest(x=x), self.assertRaisesRegex(
                ValueError,
                exact(
                    f"Expected input 'x' to have {expected_dim} coordinates, "
                    f"but got {len(x)} instead."
                ),
            ):
                log.add_new_record(x, "1234")

    def test_add_new_record_id_none_error(self):
        """Test that a ValueError is raised if the supplied job id is `None`."""

        log = SimulationsLog(self.simulations_file, input_dim=1)
        with self.assertRaisesRegex(ValueError, exact("job_id cannot be None.")):
            log.add_new_record(Input(1), None)

    def test_add_new_record_duplicate_id_error(self):
        """Test that a ValueError is raised if a record has same job id as supplied."""

        log = SimulationsLog(self.simulations_file, input_dim=1)
        log.add_new_record(Input(1), "1")

        with self.assertRaisesRegex(ValueError, exact(f"The job_id '1' is already in use.")):
            log.add_new_record(Input(1), "1")

    def test_add_new_record_job_id_different_data_types(self):
        """Test adding new records with job IDs of different data types."""

        log = SimulationsLog(self.simulations_file, 1)

        test_cases = [
            ("123", "String type should be accepted."),
            (456, "Integer type should be accepted."),
            (JobId("789"), "Custom JobId type should be accepted."),
        ]

        for job_id, message in test_cases:
            with self.subTest(job_id=job_id):
                try:
                    log.add_new_record(Input(1), job_id, JobStatus.NOT_SUBMITTED)
                except ValueError as e:
                    self.fail(f"{message} Failed with ValueError: {e}")
                except TypeError as e:
                    self.fail(f"{message} Failed with TypeError: {e}")

    def test_add_new_record_single_input(self):
        """Test that, when a record for a given input is added, the corresponding
        simulation shows up in the list of previous simulations."""

        x = Input(1)
        log = SimulationsLog(self.simulations_file, input_dim=len(x))
        log.add_new_record(x, "1234")
        self.assertEqual(((x, None),), log.get_simulations())

    def test_add_new_record_multiple_records_same_input(self):
        """Test that, when multiple records for the same input are added, one simulation
        for each record shows up in the list of previous simulations."""

        x = Input(1)
        log = SimulationsLog(self.simulations_file, input_dim=len(x))
        log.add_new_record(x, "1111")
        log.add_new_record(x, "2222")
        self.assertEqual(((x, None), (x, None)), log.get_simulations())

    def test_insert_result_missing_job_id_error(self):
        """Test that a SimulationsLogLookupError is raised if one attempts to add an
        output with a job ID that doesn't exist in the simulations log file."""

        x = Input(1)
        log = SimulationsLog(self.simulations_file, input_dim=len(x))
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

    @unittest.skip("Not sure if this is still valid?")
    def test_insert_result_multiple_job_id_error(self):
        """Test that a SimulationsLogLookupError is raised if there are multiple
        records with the same job ID when trying to insert a simulator output."""

        x = Input(1)
        log = SimulationsLog(self.simulations_file, input_dim=len(x))
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
        """Test that an empty tuple of pending jobs is returned if there are no
        records in the simulations log file."""

        log = SimulationsLog(self.simulations_file, input_dim=1)
        self.assertEqual(tuple(), log.get_pending_jobs())

    def test_get_pending_jobs_empty_when_all_completed(self):
        """Test that an empty tuple is returned if all jobs in the simulations
        log file have a simulator output."""

        log = SimulationsLog(self.simulations_file, input_dim=1)

        log.add_new_record(Input(1), job_id="1", job_status=JobStatus.COMPLETED)
        log.insert_result(job_id="1", result=10.1)

        log.add_new_record(Input(2), job_id="2", job_status=JobStatus.COMPLETED)
        log.insert_result(job_id="2", result=20.2)

        self.assertEqual(tuple(), log.get_pending_jobs())

    def test_get_pending_jobs_selects_correct_jobs(self):
        """Test that the jobs selected are those that have a Job ID but not an
        output."""

        log = SimulationsLog(self.simulations_file, input_dim=1)
        for x, job_id, y, status in (
            (Input(1), "1", 10.1, JobStatus.COMPLETED),
            (Input(2), "2", None, JobStatus.RUNNING),
            (Input(3), "3", None, JobStatus.FAILED_SUBMIT),
            (Input(4), None, None, JobStatus.FAILED),
        ):
            log.add_new_record(x, job_id, status)
            if job_id is not None:
                log.insert_result(job_id, y)

        pending_jobs = log.get_pending_jobs()

        pending_job_ids = tuple(str(job.id) for job in pending_jobs)
        self.assertEqual(("2", "3"), pending_job_ids)

    def test_get_unsubmitted_inputs_no_inputs(self):
        """Test that an empty tuple is returned if no inputs have been submitted to the
        log file."""

        log = SimulationsLog(self.simulations_file, input_dim=1)
        self.assertEqual(tuple(), log.get_unsubmitted_inputs())

    def test_get_unsubmitted_inputs_unsubmitted_input(self):
        """Test that, when an input is submitted with no job ID, it features as an
        unsubmitted input."""

        x = Input(1)
        log = SimulationsLog(self.simulations_file, input_dim=len(x))
        log.add_new_record(x)
        self.assertEqual((x,), log.get_unsubmitted_inputs())


if __name__ == "__main__":
    unittest.main()
