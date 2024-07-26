import os
import pathlib
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from numbers import Real
from threading import Thread
from time import sleep
from typing import Type
from unittest.mock import Mock, patch

from exauq.core.modelling import Input, SimulatorDomain
from exauq.sim_management.hardware import HardwareInterface, JobStatus
from exauq.sim_management.jobs import Job, JobId
from exauq.sim_management.simulators import (
    CompletedJobStrategy,
    FailedJobStrategy,
    FailedSubmitJobStrategy,
    InvalidJobStatusError,
    JobIDGenerator,
    JobManager,
    PendingCancelJobStrategy,
    PendingSubmitJobStrategy,
    RunningJobStrategy,
    SimulationsLog,
    SimulationsLogLookupError,
    Simulator,
    SubmittedJobStrategy,
    UnknownJobIdError,
)
from exauq.sim_management.types import FilePath
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

    with patch(
        "exauq.sim_management.simulators.SimulationsLog",
        new=make_fake_simulations_log_class(simulations),
    ), patch("exauq.sim_management.simulators.JobManager", new=DumbJobManager):
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

        def get_non_terminated_jobs(self):
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

        with patch(
            "exauq.sim_management.simulators.SimulationsLog",
            new=make_fake_simulations_log_class(tuple()),
        ):
            # Unix
            _ = Simulator(self.simulator_domain, self.hardware_interface, r"a/b/c")

            # Windows
            _ = Simulator(self.simulator_domain, self.hardware_interface, r"a\b\c")

            # Platform independent
            _ = Simulator(
                self.simulator_domain, self.hardware_interface, pathlib.Path("a/b/c")
            )

    def test_initialise_default_log_file(self):
        """Test that a new log file with name 'simulations.csv' is created in the
        working directory as the default."""

        with patch("exauq.sim_management.simulators.SimulationsLog") as mock:
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

        self.simulations_file.write_text(
            "Input_1,Output,Job_ID,Job_Status,Job_Level,Interface_Name\n"
        )
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
            "Input_2,Job_Status,Job_ID,Job_Level,Interface_Name,Output,Input_1\n1,Completed,0,1,server_01,2,10\n"
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

    def test_add_new_record_id_invalid_job_id_error(self):
        """Test that a ValueError is raised if the supplied job id does not define a
        numeric string."""

        log = SimulationsLog(self.simulations_file, input_dim=1)
        for job_id in [None, -1, "a1"]:
            with self.subTest(job_id=job_id):
                with self.assertRaises(ValueError):
                    log.add_new_record(Input(1), job_id)

    def test_add_new_record_duplicate_id_error(self):
        """Test that a ValueError is raised if a record has same job id as supplied."""

        log = SimulationsLog(self.simulations_file, input_dim=1)
        log.add_new_record(Input(1), "1")

        with self.assertRaisesRegex(
            ValueError, exact(f"The job_id '1' is already in use.")
        ):
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
                    log.add_new_record(Input(1), job_id, JobStatus.PENDING_SUBMIT)
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

    def test_add_new_record_default_status(self):
        """Test that, when a new record is created, it is labelled as PENDING_SUBMIT by default."""

        x1 = Input(1)
        x2 = Input(2)
        log = SimulationsLog(self.simulations_file, input_dim=len(x1))
        log.add_new_record(x1, "1")
        log.add_new_record(x2, "2")
        self.assertEqual((x1, x2), log.get_unsubmitted_inputs())

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

    def test_get_non_terminated_jobs_empty_log_file(self):
        """Test that an empty tuple of non terminated jobs is returned if there are no
        records in the simulations log file."""

        log = SimulationsLog(self.simulations_file, input_dim=1)
        self.assertEqual(tuple(), log.get_non_terminated_jobs())

    def test_get_non_terminated_jobs_empty_when_all_completed(self):
        """Test that an empty tuple is returned if all jobs in the simulations
        log file have a terminated JobStatus i.e. one of COMPLETED, FAILED, CANCELLED, FAILED_SUBMIT.
        """

        log = SimulationsLog(self.simulations_file, input_dim=1)

        log.add_new_record(Input(1), job_id="1", job_status=JobStatus.COMPLETED)
        log.insert_result(job_id="1", result=10.1)

        log.add_new_record(Input(2), job_id="2", job_status=JobStatus.COMPLETED)
        log.insert_result(job_id="2", result=20.2)

        log.add_new_record(Input(3), job_id="3", job_status=JobStatus.FAILED)

        log.add_new_record(Input(4), job_id="4", job_status=JobStatus.CANCELLED)

        log.add_new_record(Input(5), job_id="5", job_status=JobStatus.FAILED_SUBMIT)

        self.assertEqual(tuple(), log.get_non_terminated_jobs())

    def test_get_non_terminated_jobs_selects_correct_jobs(self):
        """Test that the jobs selected are those that have a valid non-terminal JobState:
        SUBMITTED, PENDING_SUBMIT or RUNNING."""

        log = SimulationsLog(self.simulations_file, input_dim=1)
        for x, job_id, y, status in (
            (Input(1), "1", 10.1, JobStatus.COMPLETED),
            (Input(2), "2", None, JobStatus.SUBMITTED),
            (Input(3), "3", None, JobStatus.PENDING_SUBMIT),
            (Input(4), "4", None, JobStatus.CANCELLED),
            (Input(5), "5", None, JobStatus.RUNNING),
            (Input(6), "6", None, JobStatus.FAILED_SUBMIT),
            (Input(7), "7", None, JobStatus.FAILED),
        ):
            log.add_new_record(x, job_id, status, interface_name="server_01")

        non_terminated_jobs = log.get_non_terminated_jobs()

        non_terminated_job_ids = tuple(str(job.id) for job in non_terminated_jobs)
        self.assertEqual(("2", "3", "5"), non_terminated_job_ids)

    def test_get_unsubmitted_inputs_no_inputs(self):
        """Test that an empty tuple is returned if no inputs have been submitted to the
        log file."""

        log = SimulationsLog(self.simulations_file, input_dim=1)
        self.assertEqual(tuple(), log.get_unsubmitted_inputs())


class TestJobManager(unittest.TestCase):
    def setUp(self):
        self.mock_simulations_log = Mock(spec=SimulationsLog)
        self.mock_simulations_log.get_non_terminated_jobs.return_value = []

        self.mock_interface1 = Mock(spec=HardwareInterface)
        self.mock_interface1.name = "mock_interface1"
        self.mock_interface1.level = 1

        self.mock_interface2 = Mock(spec=HardwareInterface)
        self.mock_interface2.name = "mock_interface2"
        self.mock_interface2.level = 2

        self.job_manager = JobManager(
            self.mock_simulations_log, [self.mock_interface1, self.mock_interface2]
        )

    def tearDown(self):
        # Ensure the JobManager is shut down properly after each test
        self.job_manager.shutdown()
        if self.job_manager._thread and self.job_manager._thread.is_alive():
            self.job_manager._thread.join(timeout=5)  # Wait up to 5 seconds

    def test_init(self):
        """Test that JobManager initializes with correct attributes."""
        self.assertIsInstance(self.job_manager._simulations_log, SimulationsLog)
        self.assertIsInstance(self.job_manager._interfaces, dict)
        self.assertEqual(len(self.job_manager._interfaces), 2)
        self.assertEqual(self.job_manager._interfaces[1][0], self.mock_interface1)
        self.assertEqual(self.job_manager._interfaces[2][0], self.mock_interface2)

    @patch("exauq.sim_management.simulators.JobIDGenerator.generate_id")
    def test_submit(self, mock_generate_id):
        """Test that a job is correctly submitted and recorded."""
        mock_generate_id.return_value = JobId("123")
        input_data = Input(1.0, 2.0)

        job = self.job_manager.submit(input_data)

        self.assertEqual(job.id, JobId("123"))
        self.assertEqual(job.data, input_data)
        self.mock_simulations_log.add_new_record.assert_called_once_with(
            input_data,
            "123",
            job_status=JobStatus.PENDING_SUBMIT,
            job_level=1,
            interface_name="mock_interface1",
        )

    def test_submit_multiple_jobs(self):
        """Test that multiple jobs can be submitted and are correctly tracked."""
        input_data1 = Input(1.0, 2.0)
        input_data2 = Input(3.0, 4.0)

        with patch(
            "exauq.sim_management.simulators.JobIDGenerator.generate_id",
            side_effect=[JobId("123"), JobId("456")],
        ):
            job1 = self.job_manager.submit(input_data1)
            job2 = self.job_manager.submit(input_data2)

        self.assertEqual(job1.id, JobId("123"))
        self.assertEqual(job2.id, JobId("456"))
        self.assertEqual(len(self.job_manager._monitored_jobs), 2)

    def test_cancel(self):
        """Test that a job can be cancelled and its status is updated."""
        job_id = JobId("123")
        mock_job = Job(job_id, Input(1.0, 2.0), 1, "mock_interface")
        self.job_manager._monitored_jobs = [mock_job]

        cancelled_job = self.job_manager.cancel(job_id)

        self.assertEqual(cancelled_job, mock_job)
        self.mock_simulations_log.update_job_status.assert_called_once_with(
            "123", JobStatus.PENDING_CANCEL
        )

    def test_cancel_non_existent_job(self):
        """Test that cancelling a non-existent job raises UnknownJobIdError."""
        non_existent_job_id = JobId("999")

        self.mock_simulations_log.get_job_status.side_effect = SimulationsLogLookupError(
            f"No such job exists with ID {non_existent_job_id}"
        )

        with self.assertRaises(UnknownJobIdError) as context:
            self.job_manager.cancel(non_existent_job_id)

        self.assertIn("Could not cancel job", str(context.exception))
        self.assertIn("no such job exists", str(context.exception))

    def test_cancel_terminated_job(self):
        """Test that cancelling a terminated job raises InvalidJobStatusError."""
        terminated_job_id = JobId("888")

        self.mock_simulations_log.get_job_status.return_value = JobStatus.COMPLETED

        with self.assertRaises(InvalidJobStatusError) as context:
            self.job_manager.cancel(terminated_job_id)

        self.assertIn("Cannot cancel 'job' with terminal status", str(context.exception))

    def test_cancel_active_job(self):
        """Test that an active job can be cancelled and its status is updated."""
        active_job_id = JobId("777")
        active_job = Job(active_job_id, Input(1.0, 2.0), 1, "mock_interface")

        self.job_manager._monitored_jobs = [active_job]

        cancelled_job = self.job_manager.cancel(active_job_id)

        self.assertEqual(cancelled_job, active_job)
        self.mock_simulations_log.update_job_status.assert_called_once_with(
            str(active_job_id), JobStatus.PENDING_CANCEL
        )

    def test_monitor(self):
        """Test that jobs are correctly added to the monitored jobs list."""
        job1 = Job(JobId("123"), Input(1.0, 2.0), 1, "mock_interface1")
        job2 = Job(JobId("456"), Input(3.0, 4.0), 1, "mock_interface1")

        self.job_manager.monitor([job1, job2])

        self.assertEqual(len(self.job_manager._monitored_jobs), 2)
        self.assertIn(job1, self.job_manager._monitored_jobs)
        self.assertIn(job2, self.job_manager._monitored_jobs)

    @patch("threading.Thread.start")
    def test_monitor_jobs_thread_creation(self, mock_thread_start):
        """Test that a monitoring thread is created when jobs are added."""
        job = Job(JobId("123"), Input(1.0, 2.0), 1, "mock_interface1")

        self.job_manager.monitor([job])

        self.assertIsNotNone(self.job_manager._thread)
        self.assertIsInstance(self.job_manager._thread, Thread)
        mock_thread_start.assert_called_once()

        self.assertIn(job, self.job_manager._monitored_jobs)

    def test_monitor_with_existing_jobs(self):
        """Test that existing non-terminated jobs are added to monitored jobs on initialization."""
        existing_job = Job(JobId("789"), Input(5.0, 6.0), 1, "mock_interface1")
        self.mock_simulations_log.get_non_terminated_jobs.return_value = [existing_job]

        new_job_manager = JobManager(
            self.mock_simulations_log, [self.mock_interface1, self.mock_interface2]
        )

        self.assertIn(existing_job, new_job_manager._monitored_jobs)

        new_job_manager.shutdown()

    def test_get_interface(self):
        """Test that the correct interface is retrieved by name."""
        interface = self.job_manager.get_interface("mock_interface1")
        self.assertEqual(interface, self.mock_interface1)

    def test_interface_selection_based_on_level(self):
        """Test that the correct interface is selected based on the specified level."""
        input_data = Input(1.0, 2.0)

        with patch(
            "exauq.sim_management.simulators.JobIDGenerator.generate_id",
            return_value=JobId("123"),
        ):
            job = self.job_manager.submit(input_data, level=2)

        self.assertEqual(job.interface_name, "mock_interface2")

    def test_remove_job(self):
        """Test that a job is correctly removed from monitored jobs and the count is updated."""
        job = Job(JobId("123"), Input(1.0, 2.0), 1, "mock_interface1")
        self.job_manager._monitored_jobs = [job]
        self.job_manager._interface_job_monitor_counts["mock_interface1"] = 1

        self.job_manager.remove_job(job)

        self.assertEqual(len(self.job_manager._monitored_jobs), 0)
        self.assertEqual(
            self.job_manager._interface_job_monitor_counts["mock_interface1"], 0
        )

    def test_shutdown(self):
        """Test that shutdown correctly stops the monitoring thread."""
        self.job_manager._thread = Mock()
        self.job_manager._thread.is_alive.return_value = True

        self.job_manager.shutdown()

        self.assertTrue(self.job_manager._shutdown_event.is_set())
        self.job_manager._thread.join.assert_called_once()


class TestJobStrategies(unittest.TestCase):
    def setUp(self):
        self.mock_job_manager = Mock(spec=JobManager)
        self.mock_job = Mock(spec=Job)
        self.mock_job.id = JobId("123")
        self.mock_job.interface_name = "mock_interface"
        self.mock_interface = Mock(spec=HardwareInterface)
        self.mock_job_manager.get_interface.return_value = self.mock_interface

    def test_completed_job_strategy(self):
        """Test that CompletedJobStrategy correctly updates job status and removes the job."""
        strategy = CompletedJobStrategy()
        self.mock_interface.get_job_output.return_value = 42.0

        strategy.handle(self.mock_job, self.mock_job_manager)

        self.mock_job_manager.simulations_log.insert_result.assert_called_once_with(
            "123", 42.0
        )
        self.mock_job_manager.simulations_log.update_job_status.assert_called_once_with(
            "123", JobStatus.COMPLETED
        )
        self.mock_job_manager.remove_job.assert_called_once_with(self.mock_job)

    def test_completed_job_strategy_no_output(self):
        """Test that CompletedJobStrategy handles jobs with no output correctly."""
        strategy = CompletedJobStrategy()
        self.mock_interface.get_job_output.return_value = None

        strategy.handle(self.mock_job, self.mock_job_manager)

        self.mock_job_manager.simulations_log.insert_result.assert_called_once_with(
            "123", None
        )
        self.mock_job_manager.simulations_log.update_job_status.assert_called_once_with(
            "123", JobStatus.COMPLETED
        )
        self.mock_job_manager.remove_job.assert_called_once_with(self.mock_job)

    def test_failed_job_strategy(self):
        """Test that FailedJobStrategy updates job status to FAILED and removes the job."""
        strategy = FailedJobStrategy()

        strategy.handle(self.mock_job, self.mock_job_manager)

        self.mock_job_manager.simulations_log.update_job_status.assert_called_once_with(
            "123", JobStatus.FAILED
        )
        self.mock_job_manager.remove_job.assert_called_once_with(self.mock_job)

    def test_running_job_strategy_from_submitted(self):
        """Test that RunningJobStrategy updates job status from SUBMITTED to RUNNING."""
        strategy = RunningJobStrategy()
        self.mock_job_manager.simulations_log.get_job_status.return_value = (
            JobStatus.SUBMITTED
        )

        strategy.handle(self.mock_job, self.mock_job_manager)

        self.mock_job_manager.simulations_log.update_job_status.assert_called_once_with(
            "123", JobStatus.RUNNING
        )

    def test_running_job_strategy_already_running(self):
        """Test that RunningJobStrategy doesn't update status for already running jobs."""
        strategy = RunningJobStrategy()
        self.mock_job_manager.simulations_log.get_job_status.return_value = (
            JobStatus.RUNNING
        )

        strategy.handle(self.mock_job, self.mock_job_manager)

        self.mock_job_manager.simulations_log.update_job_status.assert_not_called()

    def test_submitted_job_strategy(self):
        """Test that SubmittedJobStrategy doesn't perform any actions."""
        strategy = SubmittedJobStrategy()

        strategy.handle(self.mock_job, self.mock_job_manager)

        # SubmittedJobStrategy doesn't do anything, so we're just checking it doesn't raise an exception
        self.mock_job_manager.simulations_log.update_job_status.assert_not_called()
        self.mock_job_manager.remove_job.assert_not_called()

    @patch("exauq.sim_management.simulators.sleep")
    def test_pending_submit_job_strategy_success(self, mock_sleep):
        """Test that PendingSubmitJobStrategy successfully submits a job and updates its status."""
        strategy = PendingSubmitJobStrategy()

        strategy.handle(self.mock_job, self.mock_job_manager)

        self.mock_interface.submit_job.assert_called_once_with(self.mock_job)
        self.mock_job_manager.simulations_log.update_job_status.assert_called_once_with(
            "123", JobStatus.SUBMITTED
        )
        mock_sleep.assert_not_called()

    @patch("exauq.sim_management.simulators.sleep")
    @patch("random.uniform", return_value=0.05)  # To make the test deterministic
    def test_pending_submit_job_strategy_retry(self, mock_uniform, mock_sleep):
        """Test that PendingSubmitJobStrategy retries submission on failure with exponential backoff."""
        strategy = PendingSubmitJobStrategy()
        self.mock_interface.submit_job.side_effect = [
            Exception("Network error"),
            Exception("Another error"),
            None,
        ]

        strategy.handle(self.mock_job, self.mock_job_manager)

        self.assertEqual(self.mock_interface.submit_job.call_count, 3)
        self.mock_job_manager.simulations_log.update_job_status.assert_called_once_with(
            "123", JobStatus.SUBMITTED
        )
        self.assertEqual(mock_sleep.call_count, 2)
        mock_sleep.assert_any_call(2.05)
        mock_sleep.assert_any_call(4.05)

    @patch("exauq.sim_management.simulators.sleep")
    @patch("random.uniform", return_value=0.05)  # To make the test deterministic
    def test_pending_submit_job_strategy_max_retries(self, mock_uniform, mock_sleep):
        """Test that PendingSubmitJobStrategy fails after maximum retries and updates status."""
        strategy = PendingSubmitJobStrategy()
        self.mock_interface.submit_job.side_effect = Exception("Network error")

        strategy.handle(self.mock_job, self.mock_job_manager)

        self.assertEqual(self.mock_interface.submit_job.call_count, 5)  # Max retries
        self.mock_job_manager.simulations_log.update_job_status.assert_called_once_with(
            "123", JobStatus.FAILED_SUBMIT
        )
        self.assertEqual(mock_sleep.call_count, 4)  # Called 4 times for 5 attempts
        expected_sleep_times = [2.05, 4.05, 8.05, 16.05]
        for call, expected_time in zip(mock_sleep.call_args_list, expected_sleep_times):
            self.assertAlmostEqual(call[0][0], expected_time, places=2)

    def test_pending_cancel_job_strategy_success(self):
        """Test that PendingCancelJobStrategy successfully cancels a running job."""
        strategy = PendingCancelJobStrategy()
        self.mock_interface.get_job_status.return_value = JobStatus.RUNNING

        strategy.handle(self.mock_job, self.mock_job_manager)

        self.mock_interface.cancel_job.assert_called_once_with(self.mock_job.id)
        self.mock_job_manager.simulations_log.update_job_status.assert_called_once_with(
            "123", JobStatus.CANCELLED
        )
        self.mock_job_manager.remove_job.assert_called_once_with(self.mock_job)

    def test_pending_cancel_job_strategy_already_completed(self):
        """Test that PendingCancelJobStrategy raises an error for already completed jobs."""
        strategy = PendingCancelJobStrategy()
        self.mock_interface.get_job_status.return_value = JobStatus.COMPLETED

        with self.assertRaises(InvalidJobStatusError):
            strategy.handle(self.mock_job, self.mock_job_manager)

        self.mock_interface.cancel_job.assert_not_called()
        self.mock_job_manager.simulations_log.update_job_status.assert_called_once_with(
            "123", JobStatus.COMPLETED
        )
        self.mock_job_manager.remove_job.assert_called_once_with(self.mock_job)

    def test_failed_submit_job_strategy(self):
        """Test that FailedSubmitJobStrategy updates job status to FAILED_SUBMIT and removes the job."""
        strategy = FailedSubmitJobStrategy()

        strategy.handle(self.mock_job, self.mock_job_manager)

        self.mock_job_manager.simulations_log.update_job_status.assert_called_once_with(
            "123", JobStatus.FAILED_SUBMIT
        )
        self.mock_job_manager.remove_job.assert_called_once_with(self.mock_job)


class TestJobIDGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = JobIDGenerator()

    def test_generate_id_format(self):
        """Test that generated JobID has correct format and length."""
        job_id = self.generator.generate_id()
        self.assertIsInstance(job_id, JobId)
        self.assertEqual(len(str(job_id)), 17)  # YYYYMMDDHHMMSSfff format
        self.assertTrue(str(job_id).isdigit())

    def test_generate_id_uniqueness(self):
        """Test that consecutive generated JobIDs are unique."""
        job_id1 = self.generator.generate_id()
        job_id2 = self.generator.generate_id()
        self.assertNotEqual(job_id1, job_id2)

    def test_generate_id_monotonic_increase(self):
        """Test that generated JobIDs increase monotonically."""
        job_id1 = self.generator.generate_id()
        job_id2 = self.generator.generate_id()
        self.assertLess(int(str(job_id1)), int(str(job_id2)))

    def test_generate_id_millisecond_resolution(self):
        """Test that generated JobID has millisecond resolution and falls within expected time range."""
        start_time = datetime.now()
        job_id = self.generator.generate_id()
        end_time = datetime.now()

        job_id_time = datetime.strptime(str(job_id), "%Y%m%d%H%M%S%f")

        # Truncate start_time and end_time to millisecond precision
        start_time = start_time.replace(microsecond=start_time.microsecond // 1000 * 1000)
        end_time = end_time.replace(microsecond=end_time.microsecond // 1000 * 1000)

        self.assertLessEqual(start_time, job_id_time)
        self.assertLessEqual(job_id_time, end_time)

    def test_generate_id_wait_for_next_millisecond(self):
        """Test that JobIDGenerator waits for next millisecond if necessary."""
        generator = JobIDGenerator()
        job_id1 = generator.generate_id()
        sleep(0.0005)  # Sleep for half a millisecond
        job_id2 = generator.generate_id()

        # Convert JobId to datetime objects
        time1 = datetime.strptime(str(job_id1), "%Y%m%d%H%M%S%f")
        time2 = datetime.strptime(str(job_id2), "%Y%m%d%H%M%S%f")

        # Calculate the time difference in milliseconds
        time_diff_ms = (time2 - time1).total_seconds() * 1000

        # Assertions
        self.assertNotEqual(job_id1, job_id2)  # IDs should be different
        self.assertGreater(time_diff_ms, 0)  # Second ID should be later
        self.assertLess(time_diff_ms, 10)  # Difference should be less than 10ms

    def test_thread_safety(self):
        """Test that JobIDGenerator is thread-safe and generates unique IDs across multiple threads."""
        num_threads = 10
        ids_per_thread = 100
        all_ids = []

        def generate_ids():
            for _ in range(ids_per_thread):
                all_ids.append(self.generator.generate_id())

        threads = [Thread(target=generate_ids) for _ in range(num_threads)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(len(all_ids), num_threads * ids_per_thread)
        self.assertEqual(len(set(all_ids)), len(all_ids))  # All IDs are unique

    def test_generate_id_across_second_boundary(self):
        """Test that JobIDGenerator correctly handles ID generation across second boundaries."""
        generator = JobIDGenerator()

        # Get the first ID
        job_id1 = generator.generate_id()
        job_id1_str = str(job_id1)

        # Wait until we're in the next second
        job_id1_second = int(job_id1_str[12:14])
        while True:
            now = datetime.now()
            if now.second != job_id1_second:
                break
            sleep(0.01)

        # Get the second ID
        job_id2 = generator.generate_id()
        job_id2_str = str(job_id2)

        # Assert that the seconds are different
        self.assertNotEqual(job_id1_str[:14], job_id2_str[:14])  # Different seconds
        self.assertLess(int(job_id1_str), int(job_id2_str))

        # Additional check: Ensure the difference is exactly 1 second
        time_diff = datetime.strptime(
            job_id2_str[:14], "%Y%m%d%H%M%S"
        ) - datetime.strptime(job_id1_str[:14], "%Y%m%d%H%M%S")
        self.assertEqual(time_diff.total_seconds(), 1)


if __name__ == "__main__":
    unittest.main()
