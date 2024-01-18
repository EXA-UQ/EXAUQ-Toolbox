import copy
import unittest

import tests.unit.fakes as fakes
from exauq.core.designers import (
    SimpleDesigner,
    SingleLevelAdaptiveSampler,
    compute_nes_loo_error,
)
from exauq.core.emulators import MogpEmulator
from exauq.core.modelling import Input, SimulatorDomain, TrainingDatum
from tests.utilities.utilities import ExauqTestCase, exact


class TestSimpleDesigner(unittest.TestCase):
    def setUp(self) -> None:
        self.domain = SimulatorDomain([(0, 1)])
        self.designer = SimpleDesigner(self.domain)

    def test_make_design_batch_size_type_error(self):
        """Test that a TypeError is raised if something other than an int is provided
        as the size."""

        size = 2.3
        with self.assertRaisesRegex(
            TypeError,
            exact(f"Expected 'size' to be an integer but received {type(size)}."),
        ):
            self.designer.make_design_batch(size)

    def test_make_design_batch_size_negative_error(self):
        """Test that a ValueError is raised if the size provided is negative."""

        size = -1
        with self.assertRaisesRegex(
            ValueError,
            exact(
                f"Expected 'size' to be a non-negative integer but is equal to {size}."
            ),
        ):
            self.designer.make_design_batch(size)

    def test_make_design_batch_return_list_length(self):
        """Test that a list of the required size is returned."""

        for size in range(0, 3):
            with self.subTest(size=size):
                design_points = self.designer.make_design_batch(size)
                self.assertIsInstance(design_points, list)
                self.assertEqual(size, len(design_points))

    def test_make_design_batch_returns_list_inputs(self):
        """Test that a list of Input objects is returned."""

        for x in self.designer.make_design_batch(2):
            self.assertIsInstance(x, Input)

    def test_make_design_batch_returns_inputs_from_domain(self):
        """Test that the Input objects returned belong to the SimulatorDomain
        contained within the designer."""

        domain = SimulatorDomain([(2, 3), (0.5, 1)])
        designer = SimpleDesigner(domain)

        for x in designer.make_design_batch(2):
            self.assertTrue(x in domain)


class TestSingleLevelAdaptiveSampler(unittest.TestCase):
    def setUp(self) -> None:
        self.datum = TrainingDatum(Input(0), 1)
        self.initial_data = [self.datum]
        self.designer = SingleLevelAdaptiveSampler(self.initial_data)
        self.emulator = fakes.FakeGP()

    def test_training_data_empty_error(self):
        """Test that a ValueError is raised if the SLAS designer is initialised with
        an empty list of training data."""

        msg = "'initial_data' must be nonempty"
        with self.assertRaisesRegex(ValueError, exact(msg)):
            SingleLevelAdaptiveSampler([])

    def test_training_data(self):
        """Test that a SingleLevelAdaptiveSampler can be initialised from different
        collections of TrainingDatum."""

        for data in [[self.datum], tuple([self.datum])]:
            with self.subTest(data=data):
                SingleLevelAdaptiveSampler(data)

    def test_str(self):
        """Test that the string description of an instance of
        SingleLevelAdaptiveSampler designer is derived from its constituent
        parts."""

        expected = (
            "SingleLevelAdaptiveSampler designer with initial data "
            f"{str(self.initial_data)}"
        )
        self.assertEqual(expected, str(self.designer))

    def test_repr(self):
        """Test that the string representation of an instance of
        SingleLevelAdaptiveSampler designer is derived from its constituent
        parts."""

        expected = f"SingleLevelAdaptiveSampler(initial_data={repr(self.initial_data)})"
        self.assertEqual(expected, repr(self.designer))

    def test_train(self):
        """Test that training an emulator with the SLAS designer returns an
        emulator of the same type."""

        trained_emulator = self.designer.train(self.emulator)

        self.assertIsInstance(trained_emulator, type(self.emulator))

    def test_train_fits_with_initial_design(self):
        """Test that the emulator returned by the SLAS designer has been trained
        on initial data."""

        initial_design = (
            TrainingDatum(Input(0.2), 0.2),
            TrainingDatum(Input(0.55), 0.55),
        )
        designer = SingleLevelAdaptiveSampler(initial_design)

        trained_emulator = designer.train(self.emulator)

        self.assertEqual(initial_design, trained_emulator.training_data)

    def test_train_returns_new_emulator(self):
        """Test that training an emulator returns a new emulator object,
        leaving the original unchanged."""

        trained_emulator = self.designer.train(self.emulator)

        self.assertNotEqual(self.emulator, trained_emulator)
        self.assertEqual(tuple(), self.emulator.training_data)

    def test_make_design_batch_default(self):
        """Test that a list with a single Input is returned for a default batch."""

        batch = self.designer.make_design_batch(self.emulator)

        self.assertIsInstance(batch, list)
        self.assertEqual(1, len(batch))
        self.assertIsInstance(batch[0], Input)

    def test_make_design_batch_number(self):
        """Test that a list of the correct number of inputs is returned when batch
        number is specified."""

        size = 2

        batch = self.designer.make_design_batch(self.emulator, size=size)

        self.assertIsInstance(batch, list)
        self.assertEqual(size, len(batch))
        for _input in batch:
            self.assertIsInstance(_input, Input)


class TestComputeNesLooError(ExauqTestCase):
    def setUp(self) -> None:
        self.training_data = [
            TrainingDatum(Input(0, 0.2), 1),
            TrainingDatum(Input(0.3, 0.1), 2),
            TrainingDatum(Input(0.6, 0.7), 3),
            TrainingDatum(Input(0.8, 0.5), 2),
            TrainingDatum(Input(0.9, 0.9), 1),
        ]
        self.gp = MogpEmulator()
        self.gp.fit(self.training_data)

    def test_compute_nes_loo_error_arg_type_errors(self):
        """A TypeError is raised if the args are not of the correct type."""

        arg = "a"
        with self.assertRaisesRegex(
            TypeError,
            f"Expected 'gp' to be of type AbstractGaussianProcess, but received {type(arg)} instead.",
        ):
            _ = compute_nes_loo_error(arg, leave_out_idx=1)

        with self.assertRaisesRegex(
            TypeError,
            f"Expected 'leave_out_idx' to be of type int, but received {type(arg)} instead.",
        ):
            _ = compute_nes_loo_error(self.gp, leave_out_idx=arg)

    def test_compute_nes_loo_error_gives_nes_error_for_loo_gp(self):
        """The normalised expected square leave-one-out (LOO) error is equal to the
        normalised expected square error of the LOO GP for the left out training datum.
        Furthermore, the LOO GP is created with the same settings as the original GP
        and is fit with the same hyperparameters as the original GP."""

        tolerance = 1e-5
        gp_settings = {"kernel": "Matern52", "nugget": "fit"}
        gp = MogpEmulator(**gp_settings)
        gp.fit(self.training_data)
        loo_emulator = MogpEmulator(**gp_settings)
        for i, left_out_data in enumerate(self.training_data):
            remaining_data = self.training_data[:i] + self.training_data[i + 1 :]
            loo_emulator.fit(remaining_data, hyperparameters=gp.fit_hyperparameters)
            norm_err = compute_nes_loo_error(gp, i)
            self.assertEqualWithinTolerance(
                norm_err,
                loo_emulator.nes_error(left_out_data.input, left_out_data.output),
                rel_tol=tolerance,
                abs_tol=tolerance,
            )

    def test_compute_nes_loo_error_leaves_original_gp_unchanged(self):
        """The original AbstractGaussianProcess's training data and fit hyperparameters
        are unchanged after computing a NES LOO error."""

        training_data = copy.deepcopy(self.gp.training_data)
        hyperparameters = copy.deepcopy(self.gp.fit_hyperparameters)
        _ = compute_nes_loo_error(self.gp, leave_out_idx=0)

        self.assertEqual(training_data, self.gp.training_data)
        self.assertEqual(hyperparameters, self.gp.fit_hyperparameters)

    def test_compute_nes_loo_error_out_of_bounds_index_error(self):
        """A ValueError is raised if the left out index is out of the bounds of the
        emulator's training data."""

        leave_out_idx = 5
        with self.assertRaisesRegex(
            ValueError,
            f"Leave out index {leave_out_idx} is not within the bounds of the training "
            "data for 'gp'.",
        ):
            _ = compute_nes_loo_error(self.gp, leave_out_idx)

    def test_compute_nes_loo_error_no_training_data_error(self):
        """A ValueError is raised if the supplied AbstractGaussianProcess has not been
        trained on any data."""

        gp = MogpEmulator()
        with self.assertRaisesRegex(
            ValueError,
            "Cannot compute leave one out error with 'gp' because it has not been trained "
            "on data.",
        ):
            _ = compute_nes_loo_error(gp, 0)


if __name__ == "__main__":
    unittest.main()
