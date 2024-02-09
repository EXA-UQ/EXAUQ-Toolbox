import copy
import functools
import itertools
import math
import unittest
import unittest.mock
from unittest.mock import MagicMock

from scipy.stats import norm

import tests.unit.fakes as fakes
from exauq.core.designers import (
    PEICalculator,
    SimpleDesigner,
    compute_loo_errors_gp,
    compute_loo_gp,
    compute_single_level_loo_samples,
)
from exauq.core.emulators import MogpEmulator, MogpHyperparameters
from exauq.core.modelling import Input, SimulatorDomain, TrainingDatum
from exauq.core.numerics import equal_within_tolerance
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


class TestComputeLooErrorsGp(ExauqTestCase):
    def setUp(self) -> None:
        self.domain = SimulatorDomain([(0, 1)])
        self.training_data = [
            TrainingDatum(Input(0), 1),
            TrainingDatum(Input(0.5), 1),
            TrainingDatum(Input(1), 1),
        ]
        self.gp = fakes.FakeGP()
        self.gp.fit(self.training_data)

    def test_compute_loo_errors_gp_arg_type_errors(self):
        """A TypeError is raised if any of the following hold:

        * The input GP is not of type AbstractGaussianProcess.
        * The domain is not of type SimulatorDomain.
        * The supplied LOO errors GP is not None or of type AbstractGaussianProcess.
        """

        arg = "a"
        with self.assertRaisesRegex(
            TypeError,
            f"Expected 'gp' to be of type AbstractGaussianProcess, but received {type(arg)} instead.",
        ):
            _ = compute_loo_errors_gp(arg, self.domain)

        with self.assertRaisesRegex(
            TypeError,
            f"Expected 'domain' to be of type SimulatorDomain, but received {type(arg)} instead.",
        ):
            _ = compute_loo_errors_gp(self.gp, arg)

        with self.assertRaisesRegex(
            TypeError,
            f"Expected 'loo_errors_gp' to be None or of type AbstractGaussianProcess, but received {type(arg)} instead.",
        ):
            _ = compute_loo_errors_gp(self.gp, self.domain, loo_errors_gp=arg)

    def test_compute_loo_errors_domain_wrong_dim_error(self):
        """A ValueError is raised if the supplied domain's dimension does not agree with
        the dimension of the inputs in the GP's training data."""

        domain_2dim = SimulatorDomain([(0, 1), (0, 1)])
        with self.assertRaisesRegex(
            ValueError,
            "Expected all training inputs in 'gp' to belong to the domain 'domain', but this is not the case.",
        ):
            _ = compute_loo_errors_gp(self.gp, domain_2dim)

    def test_compute_loo_errors_gp_returned_gp_trainied_on_loo_errors(self):
        """The GP returned is trained on data consisting of the normalised expected square
        leave-one-out errors for each of the simulator inputs used to train the supplied
        GP."""

        loo_errors_gp = compute_loo_errors_gp(self.gp, self.domain)

        # Construct expected LOO error GP training data
        loo_errors_training_data = []
        for leave_out_idx, datum in enumerate(self.gp.training_data):
            loo_gp = compute_loo_gp(self.gp, leave_out_idx)
            loo_errors_training_data.append(
                TrainingDatum(datum.input, loo_gp.nes_error(datum.input, datum.output))
            )

        # Check actual LOO error GP training data is as expected
        self.assertTrue(
            all(
                expected.input == actual.input
                and equal_within_tolerance(expected.output, actual.output)
                for expected, actual in zip(
                    loo_errors_training_data, loo_errors_gp.training_data
                )
            )
        )

    def test_compute_loo_errors_gp_default_case_same_settings_as_input_gp(self):
        """By default, the returned GP will be constructed with the same settings used to
        initialise the input GP."""

        gp = fakes.FakeGP(predictive_mean=99)
        gp.fit(self.training_data)
        loo_errors_gp = compute_loo_errors_gp(gp, self.domain)

        self.assertEqual(gp.predictive_mean, loo_errors_gp.predictive_mean)

    def test_compute_loo_errors_gp_use_given_gp_for_returned_gp(self):
        """The returned GP will use a particular instance of an
        AbstractGaussianProcess if supplied."""

        other_gp = fakes.FakeGP(predictive_mean=99)
        loo_errors_gp = compute_loo_errors_gp(
            self.gp, self.domain, loo_errors_gp=other_gp
        )

        self.assertEqual(other_gp.predictive_mean, loo_errors_gp.predictive_mean)
        self.assertEqual(id(other_gp), id(loo_errors_gp))

    def test_compute_loo_errors_gp_leaves_original_gp_unchanged(self):
        """The original AbstractGaussianProcess's training data and fit hyperparameters
        are unchanged after computing the leave-one-out errors GP."""

        training_data = copy.deepcopy(self.gp.training_data)
        hyperparameters = copy.deepcopy(self.gp.fit_hyperparameters)

        # Default case
        loo_errors_gp = compute_loo_errors_gp(self.gp, self.domain)
        self.assertNotEqual(id(self.gp), id(loo_errors_gp))
        self.assertEqual(training_data, self.gp.training_data)
        self.assertEqual(hyperparameters, self.gp.fit_hyperparameters)

        # Case where another GP is supplied for training
        other_gp = fakes.FakeGP(predictive_mean=99)
        loo_errors_gp = compute_loo_errors_gp(
            self.gp, self.domain, loo_errors_gp=other_gp
        )
        self.assertNotEqual(id(self.gp), id(loo_errors_gp))
        self.assertEqual(training_data, self.gp.training_data)
        self.assertEqual(hyperparameters, self.gp.fit_hyperparameters)

    def test_compute_loo_errors_gp_uses_constrained_correlations_in_fitting(self):
        """The leave-one-out errors GP is trained under constrained estimation of the
        correlation length scale hyperparameters."""

        domain = SimulatorDomain([(-1, 2.5), (0, 2)])
        training_data = [
            TrainingDatum(Input(0, 0), 1),
            TrainingDatum(Input(2.1, 1), 1),
            TrainingDatum(Input(-1, 1.5), 1),
            TrainingDatum(Input(-0.3, 0.4), 1),
            TrainingDatum(Input(-0.2, 0.9), 1),
        ]
        gp = fakes.FakeGP()
        gp.fit(training_data)

        loo_gp = fakes.FakeGP()
        loo_gp = compute_loo_errors_gp(gp, domain, loo_errors_gp=fakes.FakeGP())

        scale_factor = math.sqrt(-0.5 / math.log(10 ** (-8)))
        domain_side_lengths = [
            domain.bounds[0][1] - domain.bounds[0][0],  # first dim
            domain.bounds[1][1] - domain.bounds[1][0],  # second dim
        ]
        expected_corr_bounds = [
            (scale_factor * length, None) for length in domain_side_lengths
        ]  # last entry for the predictive variance

        self.assertEqual(len(expected_corr_bounds) + 1, len(loo_gp.hyperparameter_bounds))
        self.assertEqual((None, None), loo_gp.hyperparameter_bounds[-1])
        self.assertTrue(
            all(
                equal_within_tolerance(expected[0], actual[0]) and actual[1] is None
                for expected, actual in zip(
                    expected_corr_bounds, loo_gp.hyperparameter_bounds[:-1]
                )
            )
        )

    def test_compute_loo_errors_gp_output_and_input_do_not_share_state(self):
        """In the default case, the original GP and the created LOO errors GP do not share
        any state."""

        gp = fakes.FakeGP()
        gp.foo = ["foo"]  # add an extra attribute to check for shared state with output
        gp.fit(self.training_data)
        loo_gp = compute_loo_errors_gp(gp, self.domain)

        self.assertNotEqual(id(loo_gp.foo), id(gp.foo))


class TestComputeLooGp(ExauqTestCase):
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

        # For use with FakeGP instance
        self.training_data_1dim = [
            TrainingDatum(Input(0), 1),
            TrainingDatum(Input(0.5), 1),
            TrainingDatum(Input(1), 1),
        ]

    def test_compute_loo_gp_arg_type_errors(self):
        """A TypeError is raised if any of the following hold:

        * The input GP is not of type AbstractGaussianProcess.
        * The leave-one-out index is not an integer.
        * The LOO GP is not None or of type AbstractGaussianProcess.
        """

        arg = "a"
        with self.assertRaisesRegex(
            TypeError,
            f"Expected 'gp' to be of type AbstractGaussianProcess, but received {type(arg)} instead.",
        ):
            _ = compute_loo_gp(arg, leave_out_idx=1)

        with self.assertRaisesRegex(
            TypeError,
            f"Expected 'leave_out_idx' to be of type int, but received {type(arg)} instead.",
        ):
            _ = compute_loo_gp(self.gp, leave_out_idx=arg)

        with self.assertRaisesRegex(
            TypeError,
            f"Expected 'loo_gp' to be None or of type AbstractGaussianProcess, but received {type(arg)} instead.",
        ):
            _ = compute_loo_gp(self.gp, leave_out_idx=1, loo_gp=arg)

    def test_compute_loo_gp_out_of_bounds_index_error(self):
        """A ValueError is raised if the left out index is out of the bounds of the
        input AbstractGaussianProcess's training data."""

        leave_out_idx = 5
        with self.assertRaisesRegex(
            ValueError,
            f"Leave out index {leave_out_idx} is not within the bounds of the training "
            "data for 'gp'.",
        ):
            _ = compute_loo_gp(self.gp, leave_out_idx)

    def test_compute_loo_gp_no_training_data_error(self):
        """A ValueError is raised if the supplied AbstractGaussianProcess has not been
        trained on any data."""

        gp = MogpEmulator()
        with self.assertRaisesRegex(
            ValueError,
            "Cannot compute leave one out error with 'gp' because it has not been trained "
            "on data.",
        ):
            _ = compute_loo_gp(gp, 0)

    def test_compute_loo_gp_returns_same_type_of_gp(self):
        """The return type is the same as the input GP in the default case."""

        gp = fakes.FakeGP()
        gp.fit(self.training_data_1dim)
        self.assertIsInstance(compute_loo_gp(gp, leave_out_idx=0), gp.__class__)

    def test_compute_loo_gp_trained_on_left_out_training_data_subset(self):
        """The leave-one-out Gaussian process is trained on all training data from the
        original GP except for the left out training datum."""

        for i, _ in enumerate(self.training_data):
            remaining_data = tuple(self.training_data[:i] + self.training_data[i + 1 :])
            loo_gp = compute_loo_gp(self.gp, i)
            self.assertEqual(remaining_data, loo_gp.training_data)

    def test_compute_loo_gp_uses_supplied_gp_for_loo_fitting(self):
        """If supplied, another AbstractGaussianProcess is used for fitting to the
        leave-one-out data."""

        loo_gp = fakes.FakeGP()
        loo_gp_out = compute_loo_gp(self.gp, leave_out_idx=0, loo_gp=loo_gp)
        self.assertEqual(id(loo_gp), id(loo_gp_out))
        self.assertEqual(tuple(self.training_data[1:]), loo_gp.training_data)

    def test_compute_loo_gp_fit_with_input_gp_hyperparameters(self):
        """The leave-one-out GP is fit with the same fit hyperparameters as the original
        GP."""

        # Default case
        loo_gp = compute_loo_gp(self.gp, leave_out_idx=0)
        self.assertEqual(self.gp.fit_hyperparameters, loo_gp.fit_hyperparameters)

        # Case of spplying a LOO GP directly
        loo_gp = compute_loo_gp(self.gp, leave_out_idx=0, loo_gp=fakes.FakeGP())
        self.assertEqual(self.gp.fit_hyperparameters, loo_gp.fit_hyperparameters)

    def test_compute_loo_gp_returns_gp_having_same_settings_as_input(self):
        """In the default case, the leave-one-out GP is created with the same settings
        that were used to create the input GP."""

        predictive_mean = 99
        gp = fakes.FakeGP(predictive_mean=predictive_mean)
        gp.fit(self.training_data_1dim)
        loo_emulator = compute_loo_gp(gp, leave_out_idx=0)
        self.assertEqual(gp.predictive_mean, loo_emulator.predictive_mean)

    def test_compute_loo_gp_leaves_original_gp_unchanged(self):
        """The original AbstractGaussianProcess is unchanged after computing a NES LOO
        error GP."""

        training_data = copy.deepcopy(self.gp.training_data)
        hyperparameters = copy.deepcopy(self.gp.fit_hyperparameters)

        # Default case
        _ = compute_loo_gp(self.gp, leave_out_idx=0)
        self.assertEqual(training_data, self.gp.training_data)
        self.assertEqual(hyperparameters, self.gp.fit_hyperparameters)

        # Check whether the underlying MOGP GaussianProcess has been modified by checking
        # number of training inputs.
        self.assertEqual(len(training_data), self.gp.gp.n)

        # Case of supplying a LOO GP directly
        loo_gp = copy.copy(self.gp)
        _ = compute_loo_gp(self.gp, leave_out_idx=0, loo_gp=loo_gp)
        self.assertEqual(training_data, self.gp.training_data)
        self.assertEqual(hyperparameters, self.gp.fit_hyperparameters)

        # Check whether the underlying MOGP GaussianProcess has been modified by checking
        # number of training inputs.
        self.assertEqual(len(training_data), self.gp.gp.n)

    def test_compute_loo_gp_output_not_affected_by_mutating_original(self):
        """In the default case, the original GP and the created LOO GP do not share any
        state."""

        gp = fakes.FakeGP()
        gp.foo = ["foo"]  # add an extra attribute to check for shared state with output
        gp.fit(self.training_data_1dim)
        loo_gp = compute_loo_gp(gp, leave_out_idx=0)

        self.assertNotEqual(id(loo_gp.foo), id(gp.foo))


class TestPEICalculatorInit(ExauqTestCase):
    def setUp(self):
        self.domain = SimulatorDomain([(0, 1)])
        self.training_data = [
            TrainingDatum(Input(0.1), 1),
            TrainingDatum(Input(0.3), 2),
            TrainingDatum(Input(0.5), 3),
            TrainingDatum(Input(0.7), 4),
            TrainingDatum(Input(0.9), 5),
        ]
        self.gp = MogpEmulator()

    def test_init_with_valid_parameters(self):
        """Test initialisation with valid domain and gp parameters."""
        self.gp.fit(self.training_data)

        try:
            calculator = PEICalculator(domain=self.domain, gp=self.gp)
            self.assertIsInstance(calculator, PEICalculator)
        except Exception as e:
            self.fail(f"Initialisation with valid parameters raised an exception: {e}")

    def test_init_with_invalid_domain_type(self):
        """Test initialisation with an invalid domain type and check the error message."""
        self.gp.fit(self.training_data)

        with self.assertRaises(TypeError) as context:
            PEICalculator("not_a_domain_instance", self.gp)
        self.assertIn(
            "Expected 'domain' to be of type SimulatorDomain", str(context.exception)
        )

    def test_init_with_invalid_gp_type(self):
        """Test initialisation with an invalid gp type and check the error message."""
        with self.assertRaises(TypeError) as context:
            PEICalculator(self.domain, "not_a_gp_instance")
        self.assertIn(
            "Expected 'gp' to be of type AbstractGaussianProcess", str(context.exception)
        )

    def test_validation_with_empty_gp_training_data(self):
        """Test that an error is raised if the GP training data is empty."""
        with self.assertRaises(ValueError) as context:
            PEICalculator(domain=self.domain, gp=self.gp)
        self.assertEqual("Expected 'gp' to have nonempty training data.", str(context.exception))

    def test_max_targets_with_valid_training_data(self):
        """Test that max target is calculated correctly with valid training data."""
        self.gp.fit(self.training_data)
        calculator = PEICalculator(domain=self.domain, gp=self.gp)
        self.assertEqual(calculator._max_targets, 5, "Max target should be 5")

    def test_max_targets_with_negative_values(self):
        """Test that max target is calculated correctly with negative target values."""
        negative_target_training_data = [
            TrainingDatum(Input(0.1), -1),
            TrainingDatum(Input(0.3), -2),
            TrainingDatum(Input(0.5), -3),
            TrainingDatum(Input(0.7), -4),
            TrainingDatum(Input(0.9), -5),
        ]
        self.gp.fit(negative_target_training_data)
        calculator = PEICalculator(domain=self.domain, gp=self.gp)
        self.assertEqual(calculator._max_targets, -1, "Max target should be -1")

    def test_calculate_pseudopoints_calls_domain_method(self):
        """Test to ensure wrapping functionality `_calculate_pseudopoints` correctly integrates with SimulatorDomain."""
        domain_mock = MagicMock(spec=SimulatorDomain)

        expected_pseudopoints = [(Input(1.0),), (Input(2.0),)]
        domain_mock.calculate_pseudopoints.return_value = expected_pseudopoints

        self.gp.fit(self.training_data)

        calculator = PEICalculator(domain=domain_mock, gp=self.gp)

        domain_mock.calculate_pseudopoints.assert_called_once_with(
            [datum.input for datum in self.training_data]
        )
        # Verify that _calculate_pseudopoints correctly sets the expected pseudopoints
        self.assertEqual(
            calculator._other_repulsion_points,
            expected_pseudopoints,
            "The calculated pseudopoints should match the expected return value from the domain mock.",
        )


class TestPEICalculatorExpectedImprovement(ExauqTestCase):
    def setUp(self):
        self.domain = SimulatorDomain([(0, 1), (0, 1)])
        self.training_data = [
            TrainingDatum(Input(0.1, 0.1), 1),
            TrainingDatum(Input(0.3, 0.3), 2),
            TrainingDatum(Input(0.5, 0.5), 3),
            TrainingDatum(Input(0.7, 0.7), 4),
            TrainingDatum(Input(0.9, 0.9), 5),
        ]
        self.gp = MogpEmulator()
        self.gp.fit(training_data=self.training_data)
        self.pei_calculator = PEICalculator(self.domain, self.gp)

    def test_expected_improvement_zero_std(self):
        input_point = Input(0.5, 0.5)
        ei = self.pei_calculator.expected_improvement(input_point)

        # Expected improvement should be zero due to zero standard deviation
        self.assertEqual(
            ei, 0.0, "Expected improvement should be zero for zero standard deviation."
        )

    def test_expected_improvement_positive(self):
        # Mock the GP model's predict method to return a positive expected improvement scenario
        self.gp.predict = MagicMock(
            return_value=MagicMock(estimate=6.0, standard_deviation=1.0)
        )

        input_point = Input(0.6, 0.6)
        ei = self.pei_calculator.expected_improvement(input_point)

        # Expected improvement should be positive since the estimate is greater than the max target
        self.assertGreater(
            ei,
            0.0,
            "Expected improvement should be positive for estimates greater than the max target.",
        )

    def test_expected_improvement_negative_or_zero_estimate(self):
        # Mock predict method for scenario where prediction estimate is less than the max target
        self.gp.predict = MagicMock(
            return_value=MagicMock(estimate=-1000.0, standard_deviation=1.0)
        )

        input_point = Input(0.6, 0.6)
        ei = self.pei_calculator.expected_improvement(input_point)

        # Expected improvement should be non-negative even if estimate is less than the max target
        self.assertGreaterEqual(ei, 0.0, "Expected improvement should be non-negative.")

    def test_expected_improvement_accuracy(self):
        estimate = 6.0
        standard_deviation = 2.0

        # Mock predict method
        self.gp.predict = MagicMock(
            return_value=MagicMock(
                estimate=estimate, standard_deviation=standard_deviation
            )
        )

        input_point = Input(0.5, 0.5)
        ei = self.pei_calculator.expected_improvement(input_point)

        # Manual calculation
        u = (estimate - self.pei_calculator._max_targets) / standard_deviation
        cdf_u = norm.cdf(u)
        pdf_u = norm.pdf(u)
        expected_ei = (
            estimate - self.pei_calculator._max_targets
        ) * cdf_u + standard_deviation * pdf_u

        # Assert the accuracy of the EI calculation
        self.assertEqualWithinTolerance(ei, expected_ei)

    def test_ei_at_max_target(self):
        estimate = 5.0  # Exact match to max target
        standard_deviation = 1.0  # Non-zero uncertainty

        # Configure mock
        self.gp.predict = MagicMock(
            return_value=MagicMock(
                estimate=estimate, standard_deviation=standard_deviation
            )
        )

        input_point = Input(0.5, 0.5)
        ei = self.pei_calculator.expected_improvement(input_point)

        # EI should be positive due to uncertainty
        self.assertGreater(
            ei, 0.0, "Expected improvement should be positive due to uncertainty."
        )

    def test_ei_scaling_with_std_deviation(self):
        estimate = 6.0  # Above max target
        for std_dev in [0.1, 1.0, 10.0]:  # Increasing standard deviation
            with self.subTest():
                self.gp.predict = MagicMock(
                    return_value=MagicMock(estimate=estimate, standard_deviation=std_dev)
                )

                input_point = Input(0.5, 0.5)
                ei = self.pei_calculator.expected_improvement(input_point)
                self.assertGreater(
                    ei,
                    0.0,
                    f"Expected improvement should increase with standard deviation, failed at std_dev={std_dev}.",
                )


class TestPEICalculatorRepulsion(ExauqTestCase):
    def setUp(self):
        self.domain = SimulatorDomain([(0, 1)])
        self.training_data = [
            TrainingDatum(Input(0.1), 1),
            TrainingDatum(Input(0.3), 2),
            TrainingDatum(Input(0.5), 3),
            TrainingDatum(Input(0.7), 4),
            TrainingDatum(Input(0.9), 5),
        ]
        self.pseudopoints = self.domain.calculate_pseudopoints(
            [datum.input for datum in self.training_data]
        )
        self.gp = MogpEmulator()
        self.gp.fit(training_data=self.training_data)
        self.pei_calculator = PEICalculator(self.domain, self.gp)

    def test_repulsion_factor_zero_at_training_points(self):
        # Remove other repulsion points
        self.pei_calculator._other_repulsion_points = tuple()

        for training_datum in self.training_data:
            with self.subTest():
                repulsion_factor = self.pei_calculator.repulsion(training_datum.input)
                self.assertEqual(
                    repulsion_factor, 0.0, msg="Repulsion Factor should be zero."
                )

    def test_repulsion_factor_zero_at_repulsion_points(self):
        for repulsion_point in self.pei_calculator.repulsion_points:
            with self.subTest():
                repulsion_factor = self.pei_calculator.repulsion(repulsion_point)
                self.assertEqual(
                    repulsion_factor, 0.0, msg="Repulsion Factor should be zero."
                )

    def test_repulsion_factor_formula(self):
        """Test that the repulsion factor is given by the product of terms
        (1 - correlation) for correlations between the new input and the repulsion
        points."""

        domain = SimulatorDomain([(-1, 1)])
        gp = MogpEmulator()
        training_inputs = [Input(0.2), Input(0.6)]
        training_data = [TrainingDatum(x, 1) for x in training_inputs]
        gp.fit(
            training_data,
            hyperparameters=MogpHyperparameters(
                corr_length_scales=[1], process_var=1, nugget=0
            ),
        )

        def product(numbers: list[float]) -> float:
            return functools.reduce(lambda x, y: x * y, numbers)

        x = Input(0.5)
        repulsion_pts = domain.calculate_pseudopoints([]) + tuple(training_inputs)
        expected = product([1 - gp.correlation([x], [y])[0][0] for y in repulsion_pts])
        calculator = PEICalculator(domain, gp)
        self.assertEqual(expected, calculator.repulsion(x))

    def test_invalid_input(self):
        with self.assertRaises(TypeError):
            self.pei_calculator.repulsion("invalid input")


class TestComputeSingleLevelLooSamples(ExauqTestCase):
    def setUp(self) -> None:
        self.domain = SimulatorDomain([(0, 1)])
        self.training_data = [
            TrainingDatum(Input(0.1), 1),
            TrainingDatum(Input(0.3), 2),
            TrainingDatum(Input(0.5), 3),
            TrainingDatum(Input(0.7), 4),
            TrainingDatum(Input(0.9), 5),
        ]
        self.gp = MogpEmulator()
        self.gp.fit(self.training_data)

        # A GP to return when mocking the calculation of the LOO errors GP. Uses training
        # data with the same inputs as self.gp
        self.gp_e = MogpEmulator()
        self.training_data2 = [
            TrainingDatum(datum.input, 0.1) for datum in self.training_data
        ]
        self.gp_e.fit(self.training_data2)

        # Tolerance for checking equality of new design points. Needs to be sufficiently
        # small so as to detect when two new design points are essentially the same, but
        # relaxed enough to accommodate variation coming from the stochastic nature of the
        # underlying optimsation. The following value was chosen by trial-and-error.
        self.tolerance = 1e-3

    def test_arg_type_errors(self):
        """A TypeError is raised if any of the following hold:

        * The input GP is not of type AbstractGaussianProcess.
        * The domain is not of type SimulatorDomain.
        * The batch size is not an integer.
        * The supplied LOO errors GP is not None or of type AbstractGaussianProcess.
        """

        arg = "a"
        with self.assertRaisesRegex(
            TypeError,
            exact(
                f"Expected 'gp' to be of type AbstractGaussianProcess, but received {type(arg)} instead."
            ),
        ):
            _ = compute_single_level_loo_samples(arg, self.domain)

        with self.assertRaisesRegex(
            TypeError,
            exact(
                f"Expected 'domain' to be of type SimulatorDomain, but received {type(arg)} instead."
            ),
        ):
            _ = compute_single_level_loo_samples(self.gp, arg)

        with self.assertRaisesRegex(
            TypeError,
            exact(
                f"Expected 'loo_errors_gp' to be None or of type AbstractGaussianProcess, but received {type(arg)} instead."
            ),
        ):
            _ = compute_single_level_loo_samples(self.gp, self.domain, loo_errors_gp=arg)

        with self.assertRaisesRegex(
            TypeError,
            exact(
                f"Expected 'batch_size' to be an integer, but received {type(arg)} instead."
            ),
        ):
            _ = compute_single_level_loo_samples(self.gp, self.domain, batch_size=arg)

    def test_domain_wrong_dim_error(self):
        """A ValueError is raised if the supplied domain's dimension does not agree with
        the dimension of the inputs in the GP's training data."""

        domain_2dim = SimulatorDomain([(0, 1), (0, 1)])
        with self.assertRaisesRegex(
            ValueError,
            "Expected all training inputs in 'gp' to belong to the domain 'domain', but this is not the case.",
        ):
            _ = compute_single_level_loo_samples(self.gp, domain_2dim)

    def test_non_positive_batch_size_error(self):
        """A ValueError is raised if the batch size is not a positive integer."""

        for batch_size in [0, -1]:
            with self.assertRaisesRegex(
                ValueError,
                exact(
                    f"Expected batch size to be a positive integer, but received {batch_size} instead."
                ),
            ):
                _ = compute_single_level_loo_samples(
                    self.gp, self.domain, batch_size=batch_size
                )

    def test_number_of_new_design_points_matches_batch_number(self):
        """The number of new design points returned is equal to the supplied batch
        number."""

        for batch_size in [1, 2, 3]:
            with self.subTest(batch_size=batch_size):
                design_pts = compute_single_level_loo_samples(
                    self.gp, self.domain, batch_size=batch_size
                )
                self.assertEqual(batch_size, len(design_pts))

    def test_new_design_points_differ_in_batch(self):
        """A batch of new design points consists of Input objects that are (likely)
        all distinct."""

        design_pts = compute_single_level_loo_samples(self.gp, self.domain, batch_size=2)

        self.assertFalse(
            equal_within_tolerance(
                design_pts[0],
                design_pts[1],
                rel_tol=self.tolerance,
                abs_tol=self.tolerance,
            )
        )

    def test_new_design_points_distinct_from_training_inputs(self):
        """A batch of new design points consists of Input objects that are (likely)
        distinct from the training data inputs."""

        design_pts = compute_single_level_loo_samples(self.gp, self.domain, batch_size=3)
        training_inputs = [datum.input for datum in self.training_data]

        for training_input, design_pt in itertools.product(training_inputs, design_pts):
            with self.subTest(training_input=training_input, design_pt=design_pt):
                self.assertFalse(
                    equal_within_tolerance(
                        training_input,
                        design_pt,
                        rel_tol=self.tolerance,
                        abs_tol=self.tolerance,
                    )
                )

    def test_new_design_points_lie_in_given_domain(self):
        """Each Input from a batch of design points lies in the supplied simulator
        domain."""

        domain = SimulatorDomain([(-1, 1), (1, 3.9)])
        training_data = [
            TrainingDatum(Input(0.1, 3), 1),
            TrainingDatum(Input(-0.3, 2.1), 2),
            TrainingDatum(Input(0.5, 1.9), 3),
            TrainingDatum(Input(0.7, 3.5), 4),
            TrainingDatum(Input(-0.9, 1), 5),
        ]
        gp = MogpEmulator()
        gp.fit(training_data)
        design_pts = compute_single_level_loo_samples(gp, domain, batch_size=3)
        self.assertTrue(all(design_pt in domain for design_pt in design_pts))

    def test_use_replica_of_supplied_gp_for_loo_gp(self):
        """If no AbstractGaussianProcess is specified to use as the LOO errors GP, then
        the default construction of the LOO errors GP is used instead."""

        with unittest.mock.patch(
            "exauq.core.designers.compute_loo_errors_gp", return_value=self.gp_e
        ) as mock:
            _ = compute_single_level_loo_samples(self.gp, self.domain, batch_size=1)
            mock.assert_called_once_with(self.gp, self.domain, loo_errors_gp=None)

    def test_use_supplied_loo_gp(self):
        """If an AbstractGaussianProcess is supplied to use for the LOO errors GP, then
        this is indeed used for the LOO errors GP."""

        loo_errors_gp = MogpEmulator()
        with unittest.mock.patch(
            "exauq.core.designers.compute_loo_errors_gp", return_value=self.gp_e
        ) as mock:
            _ = compute_single_level_loo_samples(
                self.gp, self.domain, batch_size=1, loo_errors_gp=loo_errors_gp
            )
            mock.assert_called_once_with(
                self.gp, self.domain, loo_errors_gp=loo_errors_gp
            )


if __name__ == "__main__":
    unittest.main()
