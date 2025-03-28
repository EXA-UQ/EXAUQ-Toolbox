import copy
import functools
import itertools
import math
import unittest
import unittest.mock
from collections.abc import Collection, Sequence
from numbers import Real
from typing import Optional
from unittest.mock import MagicMock
from warnings import catch_warnings, simplefilter

from scipy.stats import norm

import tests.unit.fakes as fakes
from exauq.core.designers import (
    PEICalculator,
    _remove_multi_level_repeated_input,
    compute_delta_coefficients,
    compute_loo_errors_gp,
    compute_loo_gp,
    compute_loo_prediction,
    compute_multi_level_loo_error_data,
    compute_multi_level_loo_errors_gp,
    compute_multi_level_loo_prediction,
    compute_multi_level_loo_samples,
    compute_single_level_loo_samples,
    compute_zero_mean_prediction,
    create_data_for_multi_level_loo_sampling,
    oneshot_lhs,
)
from exauq.core.emulators import MogpEmulator, MogpHyperparameters
from exauq.core.modelling import (
    GaussianProcessPrediction,
    Input,
    MultiLevel,
    MultiLevelGaussianProcess,
    SimulatorDomain,
    TrainingDatum,
)
from exauq.core.numerics import equal_within_tolerance
from exauq.utilities.optimisation import maximise
from tests.utilities.utilities import ExauqTestCase, exact


class TestOneshotLhs(ExauqTestCase):
    def setUp(self) -> None:
        self.domain = SimulatorDomain([(0, 1)])
        self.batch_size = 5
        self.seed = 1

    def test_oneshot_lhs_batch_size_type_error(self):
        """Test that a TypeError is raised if something other than an int is provided
        as the batch_size"""

        batch_size = 0.54
        with self.assertRaisesRegex(
            TypeError,
            exact(
                f"Expected 'batch_size' to be of type int, but received {type(batch_size)} instead."
            ),
        ):
            oneshot_lhs(self.domain, batch_size, self.seed)

    def test_oneshot_lhs_batch_size_negative_error(self):
        """Test that a ValueError is raised if the batch_size is provided as negative."""

        batch_size = -1
        with self.assertRaisesRegex(
            ValueError,
            exact(
                f"Expected 'batch_size' to be a non-negative integer >0 but is equal to {batch_size}."
            ),
        ):
            oneshot_lhs(self.domain, batch_size, self.seed)

    def test_oneshot_lhs_batch_size_zero_error(self):
        """Test that a ValueError is raised if the batch_size is provided as 0"""

        batch_size = 0
        with self.assertRaisesRegex(
            ValueError,
            exact(
                f"Expected 'batch_size' to be a non-negative integer >0 but is equal to {batch_size}."
            ),
        ):
            oneshot_lhs(self.domain, batch_size, self.seed)

    def test_oneshot_lhs_returns_tuple_inputs(self):
        """Test that a tuple of Inputs are returned"""

        for x in oneshot_lhs(self.domain, self.batch_size, self.seed):
            self.assertIsInstance(x, Input)

    def test_oneshot_lhs_return_tuple_length(self):
        """Test that length of tuple returned is correct."""

        for num_design in range(1, 4):
            with self.subTest(num_design=num_design):
                lhs_outputs = oneshot_lhs(self.domain, num_design, self.seed)
                self.assertIsInstance(lhs_outputs, tuple)
                self.assertEqual(num_design, len(lhs_outputs))

    def test_oneshot_lhs_returns_inputs_from_domain(self):
        """Test that the inputs returned belong to the SimulatorDomain provided"""

        for x in oneshot_lhs(self.domain, self.batch_size, self.seed):
            self.assertTrue(x in self.domain)


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

    def test_compute_loo_errors_gp_domain_wrong_dim_error(self):
        """A ValueError is raised if the supplied domain's dimension does not agree with
        the dimension of the inputs in the GP's training data."""

        domain_2dim = SimulatorDomain([(0, 1), (0, 1)])
        with self.assertRaisesRegex(
            ValueError,
            "Expected all training inputs in 'gp' to belong to the domain 'domain', but this is not the case.",
        ):
            _ = compute_loo_errors_gp(self.gp, domain_2dim)

    def test_compute_loo_errors_gp_returned_gp_trained_on_loo_errors(self):
        """The GP returned is trained on data consisting of the normalised expected square
        leave-one-out errors for each of the simulator inputs used to train the supplied
        GP."""

        loo_errors_gp = compute_loo_errors_gp(self.gp, self.domain)

        # Construct expected LOO error GP training data
        loo_errors_training_data = []
        for leave_out_idx, datum in enumerate(self.gp.training_data):
            loo_gp = compute_loo_gp(self.gp, leave_out_idx)
            loo_prediction = loo_gp.predict(datum.input)
            loo_errors_training_data.append(
                TrainingDatum(datum.input, loo_prediction.nes_error(datum.output))
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
            TrainingDatum(Input(0.3), 2),
            TrainingDatum(Input(0.8), 2),
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
        trained on at least 2 training data."""

        gp = MogpEmulator()
        with self.assertRaisesRegex(
            ValueError,
            "Cannot compute leave one out error with 'gp' because it has not been trained "
            "on at least 2 data points.",
        ):
            _ = compute_loo_gp(gp, 0)

        training_data = [TrainingDatum(Input(1), 1)]
        gp.fit(training_data)
        with self.assertRaisesRegex(
            ValueError,
            "Cannot compute leave one out error with 'gp' because it has not been trained "
            "on at least 2 data points.",
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


class TestPEICalculator(ExauqTestCase):
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

    def setUpFitDataOnly(self):
        self.gp.fit(self.training_data)

    def setUpPEICalculator(self):
        self.gp.fit(self.training_data)
        self.pei_calculator = PEICalculator(self.domain, self.gp)

    def test_init_with_valid_parameters(self):
        """Test initialisation with valid domain and gp parameters."""
        self.setUpFitDataOnly()

        try:
            calculator = PEICalculator(domain=self.domain, gp=self.gp)
            self.assertIsInstance(calculator, PEICalculator)
        except Exception as e:
            self.fail(f"Initialisation with valid parameters raised an exception: {e}")

    def test_init_with_invalid_domain_type(self):
        """Test initialisation with an invalid domain type and check the error message."""
        self.setUpFitDataOnly()

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
            "Expected 'gp' to be of type AbstractGaussianProcess",
            str(context.exception),
        )

    def test_validation_with_empty_gp_training_data(self):
        """Test that an error is raised if the GP training data is empty."""
        with self.assertRaises(ValueError) as context:
            PEICalculator(domain=self.domain, gp=self.gp)
        self.assertEqual(
            "Expected 'gp' to have nonempty training data.", str(context.exception)
        )

    def test_additional_repulsion_pts_not_collection_of_inputs_error(self):
        """A TypeError is raised if the additional repulsion points is not a collection
        of Input objects"""

        self.setUpFitDataOnly()

        arg = 1
        with self.assertRaisesRegex(
            TypeError,
            exact(
                f"Expected 'additional_repulsion_pts' to be of type collection of {Input}s,"
                f"but received {type(arg)} instead."
            ),
        ):
            _ = PEICalculator(self.domain, self.gp, additional_repulsion_pts=arg)

        arg2 = [Input(10), 1]
        with self.assertRaisesRegex(
            TypeError,
            exact(
                f"Expected 'additional_repulsion_pts' to be of type collection of {Input}s,"
                f"but one or more elements were of an unexpected type."
            ),
        ):
            _ = PEICalculator(self.domain, self.gp, additional_repulsion_pts=arg2)

    def test_additional_repulsion_pts_not_in_domain_error(self):
        """A ValueError is raised if any of the additional repulsion points do not belong
        to the given simulator domain."""

        self.setUpFitDataOnly()

        for bad_repulsion_pts in [[Input(1.1)], [Input(0.5), Input(0.5, 0.5)]]:
            with self.subTest(
                bad_repulsion_pts=bad_repulsion_pts
            ), self.assertRaisesRegex(
                ValueError,
                exact(
                    "Additional repulsion points must belong to simulator domain 'domain', "
                    f"but found input {bad_repulsion_pts[-1]}."
                ),
            ):
                _ = PEICalculator(
                    self.domain, self.gp, additional_repulsion_pts=bad_repulsion_pts
                )

    def test_additional_repulsion_points_included(self):
        """Inputs supplied for additional repulsion points get added to the collection of
        repulsion points for calculating pseudo-expected improvement."""

        domain = SimulatorDomain([(0, 1), (0, 1)])
        gp = fakes.WhiteNoiseGP()
        gp.fit([TrainingDatum(Input(0.1, 0.1), 1)])
        additional_repulsion_pts = [Input(0.2, 0.2), Input(0.4, 0.4)]

        pei = PEICalculator(domain, gp, additional_repulsion_pts)

        self.assertTrue(all(x in pei.repulsion_points for x in additional_repulsion_pts))

    def test_additional_repulsion_points_removes_repeats(self):
        """If an input already features in the repulsion points that would be feature in
        the PEI calculations, then it isn't added to the collection of repulsion points.
        """

        domain = SimulatorDomain([(0, 1)])
        gp = fakes.WhiteNoiseGP()
        x1 = Input(0.1)
        x2 = Input(0.2)
        gp.fit([TrainingDatum(x1, 1), TrainingDatum(x2, 1)])
        std_repulsion_points = PEICalculator(domain, gp).repulsion_points

        pei = PEICalculator(domain, gp, additional_repulsion_pts=std_repulsion_points)

        self.assertEqual(len(std_repulsion_points), len(pei.repulsion_points))

        # If the provided repulsion points have repeats, then these are only added once
        x = Input(0.5)

        pei2 = PEICalculator(domain, gp, additional_repulsion_pts=[x, x])

        self.assertEqual(len(std_repulsion_points) + 1, len(pei2.repulsion_points))

    def test_training_inputs_and_pseudopoints_not_repeated(self):
        """Given a GP with training inputs, if the domain's pseudopoints include one of
        the training inputs then this is not repeated in the repulsion points."""

        domain = SimulatorDomain([(0, 1), (0, 1)])
        training_input = Input(0, 0.5)

        # point on a domain boundary, so is a pseudopoint
        assert training_input in domain.calculate_pseudopoints([training_input])

        gp = fakes.WhiteNoiseGP()
        gp.fit([TrainingDatum(training_input, 1)])

        pei = PEICalculator(domain, gp)

        self.assertEqual(1, len([x for x in pei.repulsion_points if x == training_input]))

    def test_additional_repulsion_points_affect_repulsion_calculation(self):
        """When additional repulsion points are supplied at initialisation, these affect
        the calculation of repulsion."""

        self.setUpPEICalculator()

        # Not repulsion points
        x1, x2 = Input(0.2), Input(0.4)
        self.assertGreater(self.pei_calculator.repulsion(x1), 0)
        self.assertGreater(self.pei_calculator.repulsion(x2), 0)

        pei_calculator2 = PEICalculator(
            self.domain, self.gp, additional_repulsion_pts=[x1, x2]
        )

        self.assertEqualWithinTolerance(0, pei_calculator2.repulsion(x1))
        self.assertEqualWithinTolerance(0, pei_calculator2.repulsion(x2))

    def test_max_targets_with_valid_training_data(self):
        """Test that max target is calculated correctly with valid training data."""
        self.setUpPEICalculator()
        self.assertEqual(self.pei_calculator._max_targets, 5, "Max target should be 5")

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
        self.setUpFitDataOnly()

        domain_mock = MagicMock(spec=SimulatorDomain)

        expected_pseudopoints = [(Input(1.0),), (Input(2.0),)]
        domain_mock.calculate_pseudopoints.return_value = expected_pseudopoints

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

    def test_expected_improvement_zero_std(self):
        self.setUpPEICalculator()

        input_point = Input(0.5)
        ei = self.pei_calculator.expected_improvement(input_point)

        # Expected improvement should be zero due to zero standard deviation
        self.assertEqual(
            ei, 0.0, "Expected improvement should be zero for zero standard deviation."
        )

    def test_expected_improvement_positive(self):

        self.gp.fit(self.training_data)

        # Mock the GP model's predict method to return a positive expected improvement scenario
        self.gp.predict = MagicMock(
            return_value=MagicMock(estimate=6.0, standard_deviation=1.0)
        )

        pei = PEICalculator(self.domain, self.gp)
        input_point = Input(0.6)
        ei = pei.expected_improvement(input_point)

        # Expected improvement should be positive since the estimate is greater than the max target
        self.assertGreater(
            ei,
            0.0,
            "Expected improvement should be positive for estimates greater than the max target.",
        )

    def test_expected_improvement_negative_or_zero_estimate(self):
        self.setUpPEICalculator()

        # Mock predict method for scenario where prediction estimate is less than the max target
        self.gp.predict = MagicMock(
            return_value=MagicMock(estimate=-1000.0, standard_deviation=1.0)
        )

        input_point = Input(0.6)
        ei = self.pei_calculator.expected_improvement(input_point)

        # Expected improvement should be non-negative even if estimate is less than the max target
        self.assertGreaterEqual(ei, 0.0, "Expected improvement should be non-negative.")

    def test_expected_improvement_accuracy(self):

        self.gp.fit(self.training_data)

        # Mock predict method
        estimate = 6.0
        standard_deviation = 2.0
        self.gp.predict = MagicMock(
            return_value=MagicMock(
                estimate=estimate, standard_deviation=standard_deviation
            )
        )
        pei = PEICalculator(self.domain, self.gp)

        input_point = Input(0.6)
        ei = pei.expected_improvement(input_point)

        # Manual calculation
        u = (estimate - pei._max_targets) / standard_deviation
        cdf_u = norm.cdf(u)
        pdf_u = norm.pdf(u)
        expected_ei = (estimate - pei._max_targets) * cdf_u + standard_deviation * pdf_u

        # Assert the accuracy of the EI calculation
        self.assertEqualWithinTolerance(ei, expected_ei)

    def test_expected_improvement_at_max_target(self):

        self.gp.fit(self.training_data)

        # Configure mock
        estimate = max(datum.output for datum in self.training_data)
        standard_deviation = 1.0  # Non-zero uncertainty
        self.gp.predict = MagicMock(
            return_value=MagicMock(
                estimate=estimate, standard_deviation=standard_deviation
            )
        )

        pei = PEICalculator(self.domain, self.gp)
        input_point = Input(0.6)
        ei = pei.expected_improvement(input_point)

        # EI should be positive due to uncertainty
        self.assertGreater(
            ei, 0.0, "Expected improvement should be positive due to uncertainty."
        )

    def test_expected_improvement_scaling_with_std_deviation(self):

        self.gp.fit(self.training_data)

        estimate = 6.0  # Above max target
        for std_dev in [0.1, 1.0, 10.0]:  # Increasing standard deviation
            with self.subTest(std_dev=std_dev):
                self.gp.predict = MagicMock(
                    return_value=MagicMock(estimate=estimate, standard_deviation=std_dev)
                )

                pei = PEICalculator(self.domain, self.gp)
                input_point = Input(0.4)
                ei = pei.expected_improvement(input_point)
                self.assertGreater(
                    ei,
                    0.0,
                    f"Expected improvement should increase with standard deviation, failed at std_dev={std_dev}.",
                )

    def test_repulsion_factor_zero_at_repulsion_points(self):
        self.setUpPEICalculator()
        for repulsion_point in self.pei_calculator.repulsion_points:
            with self.subTest():
                repulsion_factor = self.pei_calculator.repulsion(repulsion_point)
                self.assertEqual(
                    repulsion_factor, 0.0, msg="Repulsion Factor should be zero."
                )

    def test_positive_repulsion_factor_positive_for_non_repulsion_inputs(self):
        self.setUpPEICalculator()
        for point in [0.2, 0.4, 0.6, 0.8]:
            with self.subTest():
                repulsion_factor = self.pei_calculator.repulsion(Input(point))
                self.assertGreater(
                    repulsion_factor, 0.0, msg="Repulsion Factor should be positive."
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
        expected = product([1 - float(gp.correlation([x], [y])) for y in repulsion_pts])
        calculator = PEICalculator(domain, gp)
        self.assertEqual(expected, calculator.repulsion(x))

    def test_add_repulsion_points_affects_repulsion_calculation(self):
        """When additional repulsion points are added, these affect the calculation of
        repulsion."""

        self.setUpPEICalculator()

        # Not repulsion points
        x1, x2 = Input(0.2), Input(0.4)
        self.assertGreater(self.pei_calculator.repulsion(x1), 0)
        self.assertGreater(self.pei_calculator.repulsion(x2), 0)

        self.pei_calculator.add_repulsion_points([x1, x2])

        self.assertEqualWithinTolerance(0, self.pei_calculator.repulsion(x1))
        self.assertEqualWithinTolerance(0, self.pei_calculator.repulsion(x2))

    def test_invalid_input(self):
        self.setUpPEICalculator()
        with self.assertRaises(TypeError):
            self.pei_calculator.repulsion("invalid input")

    def test_add_repulsion_points_multiple_inputs(self):
        """Inputs supplied for repulsion points get added to the collection of
        repulsion points for calculating pseudo-expected improvement."""

        domain = SimulatorDomain([(0, 1), (0, 1)])
        gp = fakes.WhiteNoiseGP()
        gp.fit([TrainingDatum(Input(0.1, 0.1), 1)])
        pei = PEICalculator(domain, gp)
        inputs = [Input(0.2, 0.2), Input(0.4, 0.4)]

        pei.add_repulsion_points(inputs)

        self.assertTrue(all(x in pei.repulsion_points for x in inputs))

    def test_add_repulsion_points_not_collection_of_inputs_error(self):
        """A TypeError is raised if the supplied repulsion points is not a collection
        of Input objects."""

        self.setUpPEICalculator()

        arg = 1
        with self.assertRaisesRegex(
            TypeError,
            exact(
                f"Expected 'repulsion_points' to be of type collection of {Input}s,"
                f"but received {type(arg)} instead."
            ),
        ):
            self.pei_calculator.add_repulsion_points(arg)

        arg2 = [Input(10), 1]
        with self.assertRaisesRegex(
            TypeError,
            exact(
                f"Expected 'repulsion_points' to be of type collection of {Input}s,"
                f"but one or more elements were of an unexpected type."
            ),
        ):
            self.pei_calculator.add_repulsion_points(arg2)

    def test_add_repulsion_points_not_in_domain_error(self):
        """A ValueError is raised if any of the supplied repulsion points do not belong
        to the PEI calculator's simulator domain."""

        domain = SimulatorDomain([(0, 1)])
        gp = fakes.WhiteNoiseGP()
        gp.fit([TrainingDatum(Input(0.1), 1)])
        pei = PEICalculator(domain, gp)

        for bad_repulsion_pts in [[Input(1.1)], [Input(0.5), Input(0.5, 0.5)]]:
            with self.subTest(
                bad_repulsion_pts=bad_repulsion_pts
            ), self.assertRaisesRegex(
                ValueError,
                exact(
                    f"Repulsion points must belong to the simulator domain for this {pei.__class__.__name__}, "
                    f"but found input {bad_repulsion_pts[-1]}."
                ),
            ):
                pei.add_repulsion_points(bad_repulsion_pts)

    def test_add_repulsion_points_removes_repeats(self):
        """If an input already features in the stored repulsion points, then it isn't
        added to the collection of repulsion points."""

        domain = SimulatorDomain([(0, 1)])
        gp = fakes.WhiteNoiseGP()
        x1 = Input(0.1)
        x2 = Input(0.2)
        gp.fit([TrainingDatum(x1, 1), TrainingDatum(x2, 1)])
        pei = PEICalculator(domain, gp)

        n_repulsion_pts_before = len(pei.repulsion_points)
        pei.add_repulsion_points(pei.repulsion_points)

        self.assertEqual(n_repulsion_pts_before, len(pei.repulsion_points))

        # If the provided repulsion points have repeats, then these are only added once
        x = Input(0.5)

        pei.add_repulsion_points([x, x])

        self.assertEqual(n_repulsion_pts_before + 1, len(pei.repulsion_points))

    def test_pei_calculator_invariant_to_gp_updates(self):
        """A PEI calculator initialised from a GP is not affected by updates to the GP
        (such as training the GP on new data)."""

        domain = SimulatorDomain([(0, 1), (0, 1)])
        gp = MogpEmulator()
        gp.fit(
            [
                TrainingDatum(Input(0.1, 0.1), 1),
                TrainingDatum(Input(0.3, 0.3), 2),
                TrainingDatum(Input(0.5, 0.5), 3),
                TrainingDatum(Input(0.7, 0.7), 4),
                TrainingDatum(Input(0.9, 0.9), 5),
            ]
        )

        pei = PEICalculator(domain, gp)

        # Choose an input far away from data to get interesting variation in repulsion,
        # expected improvement and PEI.
        x = Input(0.1, 0.9)
        repulsion_points_before = pei.repulsion_points
        repulsion_before = pei.repulsion(x)
        ei_before = pei.expected_improvement(x)
        pei_before = pei.compute(x)

        # Now modify the GP by training it on new data
        gp.fit(
            [
                TrainingDatum(Input(0.2, 0.2), 0.8),
                TrainingDatum(Input(0.4, 0.4), 1.3),
                TrainingDatum(Input(0.6, 0.6), -9.9),
                TrainingDatum(Input(0.8, 0.8), -0.54),
                TrainingDatum(Input(1, 1), 50),
            ]
        )

        # Check outputs of public methods / properties haven't changed
        self.assertEqual(repulsion_points_before, pei.repulsion_points)
        self.assertEqual(repulsion_before, pei.repulsion(x))
        self.assertEqual(ei_before, pei.expected_improvement(x))
        self.assertEqual(pei_before, pei.compute(x))


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
        * The additional repulsion points is not a collection of Input objects.
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
                f"Expected 'batch_size' to be of type int, but received {type(arg)} instead."
            ),
        ):
            _ = compute_single_level_loo_samples(self.gp, self.domain, batch_size=arg)

        arg2 = 1
        with self.assertRaisesRegex(
            TypeError,
            exact(
                f"Expected 'additional_repulsion_pts' to be of type collection of {Input}s,"
                f"but received {type(arg2)} instead."
            ),
        ):
            _ = compute_single_level_loo_samples(
                self.gp, self.domain, additional_repulsion_pts=arg2
            )

        arg3 = [Input(10), 1]
        with self.assertRaisesRegex(
            TypeError,
            exact(
                f"Expected 'additional_repulsion_pts' to be of type collection of {Input}s,"
                "but one or more elements were of an unexpected type."
            ),
        ):
            _ = compute_single_level_loo_samples(
                self.gp, self.domain, additional_repulsion_pts=arg3
            )

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

    def test_additional_repulsion_pts_not_in_domain_error(self):
        """A ValueError is raised if any of the additional repulsion points do not belong
        to the given simulator domain."""

        domain = SimulatorDomain([(0, 1)])
        for bad_repulsion_pts in [[Input(1.1)], [Input(0.5), Input(0.5, 0.5)]]:
            with self.subTest(
                bad_repulsion_pts=bad_repulsion_pts
            ), self.assertRaisesRegex(
                ValueError,
                exact(
                    "Additional repulsion points must belong to simulator domain 'domain', "
                    f"but found input {bad_repulsion_pts[-1]}."
                ),
            ):
                _ = compute_single_level_loo_samples(
                    self.gp, domain, additional_repulsion_pts=bad_repulsion_pts
                )

    def test_unseeded_pei_maximisation_default(self):
        """The calculation of new design points involves unseeded maximisation of
        pseudo-expected improvement by default."""

        mock_maximise_return = (self.domain.scale([0.5]), 1)
        with unittest.mock.patch(
            "exauq.core.designers.maximise",
            autospec=True,
            return_value=mock_maximise_return,
        ) as mock_maximise:
            _ = compute_single_level_loo_samples(self.gp, self.domain)

        # checks {"seed": None} is a subset of mock_maximise.call_args.kwargs
        self.assertLessEqual(
            {"seed": None}.items(), mock_maximise.call_args.kwargs.items()
        )

    def test_use_of_seed_sequence(self):
        """If a seed is provided with a batch_size > 1, then maximisation of pseudo-expected improvement is
        performed with the newly generated sequence of seeds which should all be different.
        """

        mock_maximise_return = (self.domain.scale([0.5]), 1)
        seed = 99
        batch_size = 5
        with unittest.mock.patch(
            "exauq.core.designers.maximise",
            autospec=True,
            return_value=mock_maximise_return,
        ) as mock_maximise:
            _ = compute_single_level_loo_samples(
                self.gp, self.domain, batch_size=batch_size, seed=seed
            )

        # collect all the seeds used
        seeds_used = [
            call.kwargs["seed"]
            for call in mock_maximise.call_args_list
            if "seed" in call.kwargs
        ]

        # checks all seeds are different
        self.assertTrue(len(seeds_used) == len(set(seeds_used)))

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

    def test_additional_repulsion_pts_used_for_pseudo_expected_improvement(self):
        """If additional repulsion points are provided, then these are used in the
        calculation of pseudo-expected improvement for the LOO errors GP."""

        # Compute a new design point
        design_pts = compute_single_level_loo_samples(self.gp, self.domain, seed=1)

        # Re-run computation but now using the new design point as a repulsion point.
        # Should find a different design point created.
        design_pts2 = compute_single_level_loo_samples(
            self.gp, self.domain, additional_repulsion_pts=design_pts, seed=1
        )

        self.assertNotEqualWithinTolerance(design_pts2[0], design_pts[0])

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


class TestComputeMultiLevelLooPrediction(ExauqTestCase):
    def setUp(self) -> None:
        self.training_data = MultiLevel(
            {
                1: (TrainingDatum(Input(0.1), 1), TrainingDatum(Input(0.2), 1)),
                2: (TrainingDatum(Input(0.3), 1), TrainingDatum(Input(0.4), 1)),
                3: (TrainingDatum(Input(0.5), 1), TrainingDatum(Input(0.6), 1)),
            }
        )

    @staticmethod
    def compute_loo_prediction_level_term(
        mlgp: MultiLevelGaussianProcess, level: int, leave_out_idx: int
    ):
        # Get left-out training data
        loo_input = mlgp[level].training_data[leave_out_idx].input
        loo_output = mlgp[level].training_data[leave_out_idx].output

        # Make leave-one-out prediction at supplied level
        loo_prediction = compute_loo_gp(mlgp[level], leave_out_idx).predict(loo_input)

        # Get mean and variance contributions at supplied level
        mean_at_level = loo_prediction.estimate - loo_output
        variance_at_level = loo_prediction.variance

        return GaussianProcessPrediction(mean_at_level, variance_at_level)

    @staticmethod
    def compute_loo_prediction_other_levels_term(
        mlgp: MultiLevelGaussianProcess,
        level: int,
        leave_out_idx: int,
    ) -> list[GaussianProcessPrediction]:
        # Get left-out training inputs
        loo_input = mlgp[level].training_data[leave_out_idx].input

        # Get mean and variance contributions at other levels
        other_level_predictions = [
            compute_zero_mean_prediction(mlgp[j], loo_input)
            for j in mlgp.levels
            if not j == level
        ]
        return [
            GaussianProcessPrediction(
                prediction.estimate,
                prediction.variance,
            )
            for prediction in other_level_predictions
        ]

    def test_invalid_leave_out_idx_error(self):
        """A ValueError is raised if the leave-out index is out of the range of indices
        for training data at the specified level."""

        training_data = MultiLevel(
            {
                1: (TrainingDatum(Input(0.1), 1), TrainingDatum(Input(0.2), 1)),
                2: (TrainingDatum(Input(0.3), 1),),
                3: (TrainingDatum(Input(0.5), 1), TrainingDatum(Input(0.6), 1)),
            }
        )

        mlgp = MultiLevelGaussianProcess([fakes.WhiteNoiseGP() for _ in training_data])
        mlgp.fit(training_data)

        level = 2
        for leave_out_idx in [-1, 1]:
            with self.subTest(leave_out_idx=leave_out_idx):
                with self.assertRaisesRegex(
                    ValueError,
                    exact(
                        "'leave_out_idx' should define a zero-based index for the training data "
                        f"of length {len(training_data[level])} at level {level}, but received "
                        f"out of range index {leave_out_idx}."
                    ),
                ):
                    _ = compute_multi_level_loo_prediction(mlgp, level, leave_out_idx)

    def test_intersecting_training_inputs_error(self):
        """A ValueError is raised if a training input at some level also appears as a
        training input in another level, for the supplied multi-level GP."""

        # Test for two different training datasets.

        # Training dataset 1
        repeated_input1 = Input(0.1)
        problem_level1, problem_level2 = 1, 3
        training_data1 = MultiLevel(
            {
                problem_level1: (
                    TrainingDatum(repeated_input1, 1),
                    TrainingDatum(Input(0.2), 1),
                ),
                2: (TrainingDatum(Input(0.3), 1), TrainingDatum(Input(0.4), 1)),
                problem_level2: (
                    TrainingDatum(Input(0.5), 1),
                    TrainingDatum(repeated_input1, 1),
                ),
            }
        )

        mlgp = MultiLevelGaussianProcess([fakes.WhiteNoiseGP() for _ in training_data1])
        mlgp.fit(training_data1)

        # This check is independent of the level or left-out index
        for level in mlgp.levels:
            for leave_out_idx in range(len(mlgp.training_data[level])):
                with self.subTest(
                    level=level, leave_out_idx=leave_out_idx
                ), self.assertRaisesRegex(
                    ValueError,
                    exact(
                        "Training inputs across all levels must be distinct, but found "
                        f"common input {repeated_input1} at levels {problem_level1}, {problem_level2}."
                    ),
                ):
                    _ = compute_multi_level_loo_prediction(mlgp, level, leave_out_idx)

        # Training dataset 2
        repeated_input2 = Input(0.3)
        problem_level1, problem_level2 = 2, 3
        training_data2 = MultiLevel(
            {
                1: (TrainingDatum(Input(0.1), 1), TrainingDatum(Input(0.2), 1)),
                problem_level1: (
                    TrainingDatum(repeated_input2, 1),
                    TrainingDatum(Input(0.4), 1),
                ),
                problem_level2: (
                    TrainingDatum(Input(0.5), 1),
                    TrainingDatum(Input(0.6), 1),
                    TrainingDatum(repeated_input2, 1),
                ),
            }
        )

        mlgp2 = MultiLevelGaussianProcess([fakes.WhiteNoiseGP() for _ in training_data2])
        mlgp2.fit(training_data2)

        # This check is independent of the level or left-out index
        for level in mlgp2.levels:
            for leave_out_idx in range(len(mlgp2.training_data[level])):
                with self.subTest(
                    level=level, leave_out_idx=leave_out_idx
                ), self.assertRaisesRegex(
                    ValueError,
                    exact(
                        "Training inputs across all levels must be distinct, but found "
                        f"common input {repeated_input2} at levels {problem_level1}, {problem_level2}."
                    ),
                ):
                    _ = compute_multi_level_loo_prediction(mlgp2, level, leave_out_idx)

    def test_prediction_for_single_level(self):
        """If a multi-level GP with only one level is supplied, then the returned
        prediction corresponds to the single-level leave-one-out prediction."""

        training_data = MultiLevel(
            {
                1: (TrainingDatum(Input(0.1), 1), TrainingDatum(Input(0.2), 1)),
            }
        )

        coefficient = 10
        mlgp = MultiLevelGaussianProcess(
            [fakes.WhiteNoiseGP()], coefficients=[coefficient]
        )
        mlgp.fit(training_data)

        level, leave_out_idx = 1, 0
        term = self.compute_loo_prediction_level_term(mlgp, level, leave_out_idx)
        expected = GaussianProcessPrediction(
            estimate=(coefficient * term.estimate),
            variance=((coefficient**2) * term.variance),
        )
        self.assertEqual(
            expected,
            compute_multi_level_loo_prediction(mlgp, level, leave_out_idx),
        )

    def test_prediction_combination_loo_prediction_at_level_and_at_other_levels(self):
        """The overall multi-level leave-one-out (LOO) prediction for a given level is a
        sum of a term for the level and a term for the other levels."""

        mlgp = MultiLevelGaussianProcess(
            [
                fakes.WhiteNoiseGP(prior_mean=level, noise_level=(11 * level))
                for level in self.training_data
            ]
        )
        mlgp.fit(self.training_data)

        for level in mlgp.levels:
            for leave_out_idx, _ in enumerate(mlgp.training_data[level]):
                level_term = self.compute_loo_prediction_level_term(
                    mlgp, level, leave_out_idx
                )
                other_terms = self.compute_loo_prediction_other_levels_term(
                    mlgp, level, leave_out_idx
                )

                expected = GaussianProcessPrediction(
                    level_term.estimate + sum(term.estimate for term in other_terms),
                    level_term.variance + sum(term.variance for term in other_terms),
                )

                loo_prediction = compute_multi_level_loo_prediction(
                    mlgp, level, leave_out_idx
                )
                self.assertEqual(expected, loo_prediction)

    def test_sum_weighted_by_coefficients(self):
        """The overall multi-level leave-one-out (LOO) prediction for a given level is a
        weighted sum of a terms, with the weights coming from the coefficients for the
        multi-level GP."""

        coefficients = [1, 10, 100]
        mlgp = MultiLevelGaussianProcess(
            [fakes.WhiteNoiseGP() for _ in self.training_data],
            coefficients=coefficients,
        )
        mlgp.fit(self.training_data)

        for level in mlgp.levels:
            for leave_out_idx, _ in enumerate(mlgp.training_data[level]):
                level_term = self.compute_loo_prediction_level_term(
                    mlgp, level, leave_out_idx
                )
                other_terms = self.compute_loo_prediction_other_levels_term(
                    mlgp, level, leave_out_idx
                )
                all_terms = (
                    other_terms[: level - 1] + [level_term] + other_terms[level - 1 :]
                )

                expected = GaussianProcessPrediction(
                    estimate=sum(
                        coeff * term.estimate
                        for coeff, term in zip(coefficients, all_terms)
                    ),
                    variance=sum(
                        (coeff**2) * term.variance
                        for coeff, term in zip(coefficients, all_terms)
                    ),
                )

                loo_prediction = compute_multi_level_loo_prediction(
                    mlgp, level, leave_out_idx
                )
                self.assertEqual(expected, loo_prediction)


class TestComputeLooPrediction(ExauqTestCase):
    def test_calculation_mean_and_variance(self):
        training_data = (TrainingDatum(Input(0.1), 1), TrainingDatum(Input(0.2), -1))
        prior_mean = 10
        noise_level = 100  # same as process variance
        gp = fakes.WhiteNoiseGP(prior_mean, noise_level)

        gp.fit(training_data)

        for leave_out_idx, datum in enumerate(training_data):
            prediction_level_term = compute_loo_prediction(gp, leave_out_idx)

            # For White Noise GP, LOO mean is the prior mean and LOO variance is the noise
            # level.
            expected_term = GaussianProcessPrediction(
                estimate=(prior_mean - datum.output),
                variance=noise_level,
            )
            self.assertEqual(expected_term, prediction_level_term)


class TestComputeZeroMeanPrediction(ExauqTestCase):
    def test_no_training_data_error(self):
        """A ValueError is raised if the supplied GP has not been trained on any data."""

        gp = fakes.WhiteNoiseGP()

        with self.assertRaisesRegex(
            ValueError,
            exact(
                "Cannot calculate zero-mean prediction: 'gp' hasn't been trained on any data."
            ),
        ):
            _ = compute_zero_mean_prediction(gp, Input(1))

    def test_calculation_mean_and_variance(self):
        """The mean of the returned prediction at a given input ``x`` is given by
        ``cov(x) * K_inv * y`` where  ``cov(x)`` is the covariance matrix at the point
        ``x``, ``K_inv`` is the inverse of the covariance matrix for the GP's training
        data and ``y`` is the vector of simulator outputs in the training data. The
        variance of the returned prediction is equal to the predictive variance of the GP
        at ``x``."""

        training_data = (TrainingDatum(Input(0.1), 1), TrainingDatum(Input(0.2), -1))
        prior_mean = 10
        noise_level = 100  # same as process variance
        gp = fakes.WhiteNoiseGP(prior_mean, noise_level)

        gp.fit(training_data)

        # Case where new input not in training data
        x = Input(0.3)
        prediction = compute_zero_mean_prediction(gp, x)
        self.assertEqual(
            GaussianProcessPrediction(
                estimate=0,  # because correlation = 0 at new point (for White Noise GP)
                variance=noise_level,
            ),
            prediction,
        )

        # Cases where new input is in training data
        for datum in training_data:
            x = datum.input
            y = datum.output
            prediction = compute_zero_mean_prediction(gp, x)
            self.assertEqual(
                GaussianProcessPrediction(
                    estimate=y,  # because kernel = 1 at x and zero elsewhere (for White Noise GP)
                    variance=0,  # because estimating at training input
                ),
                prediction,
            )


class TestComputeMultiLevelLooErrorData(ExauqTestCase):
    def setUp(self) -> None:
        self.domain = SimulatorDomain([(0, 1)])
        self.training_data = MultiLevel(
            {
                1: (TrainingDatum(Input(0.1), 1), TrainingDatum(Input(0.2), 1)),
                2: (TrainingDatum(Input(0.3), 1), TrainingDatum(Input(0.4), 1)),
                3: (TrainingDatum(Input(0.5), 1), TrainingDatum(Input(0.6), 1)),
            }
        )
        self.mlgp = MultiLevelGaussianProcess(
            [fakes.WhiteNoiseGP() for _ in self.training_data]
        )
        self.mlgp.fit(self.training_data)

    def test_calculation_of_loo_prediction_errors(self):
        """Each level of the returned data consists of normalised estimated square errors
        of multi-level leave-one-out predictions."""

        ml_loo_error_data = compute_multi_level_loo_error_data(self.mlgp)

        for level, loo_error_data in ml_loo_error_data.items():
            training_inputs = [datum.input for datum in self.mlgp.training_data[level]]
            for leave_out_idx, loo_error_datum in enumerate(loo_error_data):
                loo_prediction = compute_multi_level_loo_prediction(
                    self.mlgp, level, leave_out_idx
                )
                expected = TrainingDatum(
                    training_inputs[leave_out_idx], loo_prediction.nes_error(0)
                )
                self.assertEqual(expected, loo_error_datum)

    def test_single_training_datum_error(self):
        """A ValueError is raised if there are any levels in the supplied multi-level GP
        only have 1 item of training data."""

        training_data = MultiLevel(
            {
                1: (TrainingDatum(Input(0.1), 1), TrainingDatum(Input(0.2), 1)),
                2: (TrainingDatum(Input(0.3), 1),),
                3: (TrainingDatum(Input(0.6), 1),),
            }
        )
        bad_levels = ", ".join(
            sorted({str(level) for level, data in training_data.items() if len(data) < 2})
        )
        mlgp = MultiLevelGaussianProcess([fakes.WhiteNoiseGP() for _ in training_data])
        mlgp.fit(training_data)

        with self.assertRaisesRegex(
            ValueError,
            exact(
                f"Could not perform leave-one-out calculation: levels {bad_levels} not trained on at "
                "least two training data."
            ),
        ):
            _ = compute_multi_level_loo_error_data(mlgp)


class TestComputeMultiLevelLooErrorsGp(ExauqTestCase):
    def setUp(self) -> None:
        self.domain = SimulatorDomain([(0, 1)])
        self.training_data = MultiLevel(
            {
                1: (TrainingDatum(Input(0.1), 1), TrainingDatum(Input(0.2), 1)),
                2: (TrainingDatum(Input(0.3), 1), TrainingDatum(Input(0.4), 1)),
                3: (TrainingDatum(Input(0.5), 1), TrainingDatum(Input(0.6), 1)),
            }
        )
        self.mlgp = MultiLevelGaussianProcess(
            [fakes.WhiteNoiseGP() for _ in self.training_data]
        )
        self.mlgp.fit(self.training_data)

    def test_incompatible_levels_error(self):
        """A ValueError is raised if the levels for the supplied multi-level GPs are
        not all the same."""

        other_mlgp = MultiLevelGaussianProcess(
            {
                1: fakes.WhiteNoiseGP(),
                3: fakes.WhiteNoiseGP(),
                4: fakes.WhiteNoiseGP(),
            }
        )

        with self.assertRaisesRegex(
            ValueError,
            exact(
                f"Expected the levels {other_mlgp.levels} of 'output_mlgp' to match the levels "
                f"{self.mlgp.levels} from 'mlgp'."
            ),
        ):
            _ = compute_multi_level_loo_errors_gp(
                self.mlgp, self.domain, output_mlgp=other_mlgp
            )

    def test_returned_multi_level_gp_trained_on_loo_errors(self):
        """The multi-level GP returned is trained on multi-level data consisting
        of the normalised expectation squared leave-one-out errors for each of the
        training inputs, at each level."""

        errors_gp = compute_multi_level_loo_errors_gp(self.mlgp, self.domain)
        self.assertEqual(
            compute_multi_level_loo_error_data(self.mlgp), errors_gp.training_data
        )

    def test_default_case_same_settings_as_input_gp(self):
        """By default, the returned multi-level GP will be constructed with the same
        settings used to initialise the input multi-level GP."""

        means = [10, 20, 30]
        noise_levels = [1, 2, 3]
        mlgp = MultiLevelGaussianProcess(
            [fakes.WhiteNoiseGP(mean, noise) for mean, noise in zip(means, noise_levels)]
        )
        mlgp.fit(self.training_data)

        errors_gp = compute_multi_level_loo_errors_gp(mlgp, self.domain)

        for level in mlgp.levels:
            mlgp_params = (mlgp[level].prior_mean, mlgp[level].noise_level)
            errors_gp_params = (
                errors_gp[level].prior_mean,
                errors_gp[level].noise_level,
            )
            self.assertEqual(mlgp_params, errors_gp_params)

    def test_use_given_mlgp_for_returned_mlgp(self):
        """The returned multi-level GP will use a particular multi-level GP if supplied."""

        other_mlgp = MultiLevelGaussianProcess(
            [fakes.WhiteNoiseGP(99, 100) for _ in self.training_data]
        )

        errors_gp = compute_multi_level_loo_errors_gp(
            self.mlgp, self.domain, output_mlgp=other_mlgp
        )

        self.assertIs(errors_gp, other_mlgp)

    def test_leaves_original_gp_unchanged(self):
        """The original multi-level's training data and fit hyperparameters are
        unchanged."""

        training_data = copy.deepcopy(self.mlgp.training_data)
        hyperparameters = copy.deepcopy(self.mlgp.fit_hyperparameters)

        # Default case
        loo_errors_gp = compute_multi_level_loo_errors_gp(self.mlgp, self.domain)
        self.assertIsNot(self.mlgp, loo_errors_gp)
        self.assertEqual(training_data, self.mlgp.training_data)
        self.assertEqual(hyperparameters, self.mlgp.fit_hyperparameters)

    def test_output_and_input_multi_level_gps_do_not_share_state(self):
        """In the default case, the original multi-level GP and the created multi-level
        LOO errors GP do not share any state."""

        self.mlgp.foo = [
            "foo"
        ]  # add an extra attribute to check for shared state with output

        errors_gp = compute_multi_level_loo_errors_gp(self.mlgp, self.domain)

        self.assertIsNot(errors_gp.foo, self.mlgp.foo)


class TestComputeMultiLevelLooSamples(ExauqTestCase):
    @staticmethod
    def make_level_costs(costs: Sequence[Real]) -> MultiLevel[Real]:
        return MultiLevel(costs)

    @staticmethod
    def get_levels(costs: MultiLevel[Real]) -> set[int]:
        return set(costs.levels)

    def setUp(self) -> None:
        self.tolerance = 1e-3
        self.default_domain = SimulatorDomain([(0, 1)])
        gp1 = MogpEmulator()
        gp1.fit(
            [
                TrainingDatum(Input(0.2), 1),
                TrainingDatum(Input(0.4), 1),
                TrainingDatum(Input(0.6), 1),
            ]
        )
        gp2 = MogpEmulator()
        gp2.fit(
            [
                TrainingDatum(Input(0.3), 2),
                TrainingDatum(Input(0.5), -2),
                TrainingDatum(Input(0.7), 2),
            ]
        )
        self.default_mlgp = MultiLevelGaussianProcess([gp1, gp2])
        self.default_costs = MultiLevel([1, 11])

    def compute_multi_level_loo_samples(
        self,
        mlgp: Optional[MultiLevelGaussianProcess] = None,
        domain: Optional[SimulatorDomain] = None,
        costs: Optional[MultiLevel[Real]] = None,
        batch_size: Optional[int] = 1,
        additional_repulsion_pts: Optional[MultiLevel[Collection[Input]]] = None,
        seeds: Optional[MultiLevel[int]] = None,
    ) -> MultiLevel[tuple[Input]]:
        mlgp = self.default_mlgp if mlgp is None else mlgp
        domain = self.default_domain if domain is None else domain
        costs = self.default_costs if costs is None else costs

        return compute_multi_level_loo_samples(
            mlgp,
            domain,
            costs,
            batch_size=batch_size,
            additional_repulsion_pts=additional_repulsion_pts,
            seeds=seeds,
        )

    def test_arg_type_errors(self):
        """A TypeError is raised if any of the following hold:

        * The input multi-level GP is not of type MultiLevelGaussianProcess.
        * The domain is not of type SimulatorDomain.
        * The batch size is not an integer.
        * The repulsion points are not a MultiLevel Collection of Inputs (or None)
        * The seeds are not a MultiLevel collection (or None).
        """

        arg = "a"
        with self.assertRaisesRegex(
            TypeError,
            exact(
                f"Expected 'mlgp' to be of type {MultiLevelGaussianProcess}, but received {type(arg)} instead."
            ),
        ):
            _ = self.compute_multi_level_loo_samples(mlgp=arg)

        with self.assertRaisesRegex(
            TypeError,
            exact(
                f"Expected 'domain' to be of type {SimulatorDomain}, but received {type(arg)} instead."
            ),
        ):
            _ = self.compute_multi_level_loo_samples(domain=arg)

        with self.assertRaisesRegex(
            TypeError,
            exact(
                f"Expected 'batch_size' to be of type int, but received {type(arg)} instead."
            ),
        ):
            _ = self.compute_multi_level_loo_samples(batch_size=arg)

        with self.assertRaisesRegex(
            TypeError,
            exact(
                f"Expected 'additional_repulsion_pts' to be of type MultiLevel collection of {Input}s, "
                f"but received {type(arg)} instead."
            ),
        ):
            _ = self.compute_multi_level_loo_samples(additional_repulsion_pts=arg)

        with self.assertRaisesRegex(
            TypeError,
            exact(
                f"Expected 'seeds' to be of type {MultiLevel} of int, but "
                f"received {type(arg)} instead."
            ),
        ):
            _ = self.compute_multi_level_loo_samples(seeds=arg)

    def test_domain_wrong_dim_error(self):
        """A ValueError is raised if the supplied domain's dimension does not agree with
        the dimension of the inputs in the GP's training data."""

        domain_2dim = SimulatorDomain([(0, 1), (0, 1)])
        with self.assertRaisesRegex(
            ValueError,
            "Expected all training inputs in 'mlgp' to belong to the domain 'domain', but this is not the case.",
        ):
            _ = self.compute_multi_level_loo_samples(domain=domain_2dim)

    def test_non_positive_batch_size_error(self):
        """A ValueError is raised if the batch size is not a positive integer."""

        for batch_size in [0, -1]:
            with self.assertRaisesRegex(
                ValueError,
                exact(
                    f"Expected batch size to be a positive integer, but received {batch_size} instead."
                ),
            ):
                _ = self.compute_multi_level_loo_samples(batch_size=batch_size)

    def test_differing_input_arg_levels_error(self):
        """A ValueError is raised if the levels found in the multi-level Gaussian
        process do not also appear in the simulator costs and seeds (if provided)."""

        costs = self.make_level_costs([1])
        missing_level = (set(self.default_mlgp.levels) - set(costs.levels)).pop()
        with self.assertRaisesRegex(
            ValueError,
            f"Level {missing_level} from 'mlgp' does not have associated level from 'costs'.",
        ):
            self.compute_multi_level_loo_samples(costs=costs)

        seeds = MultiLevel([1])
        missing_level = (set(self.default_mlgp.levels) - set(seeds.levels)).pop()
        with self.assertRaisesRegex(
            ValueError,
            f"Level {missing_level} from 'mlgp' does not have associated level from 'seeds'.",
        ):
            self.compute_multi_level_loo_samples(seeds=seeds)

    def test_intersecting_training_inputs_error(self):
        """A ValueError is raised if a training input at some level also appears as a
        training input in another level, for the supplied multi-level GP."""

        domain = SimulatorDomain([(0, 1)])
        repeated_input1 = Input(0.1)
        problem_level1, problem_level2 = 1, 3
        training_data1 = MultiLevel(
            {
                problem_level1: (
                    TrainingDatum(repeated_input1, 1),
                    TrainingDatum(Input(0.2), 1),
                ),
                2: (TrainingDatum(Input(0.3), 1), TrainingDatum(Input(0.4), 1)),
                problem_level2: (
                    TrainingDatum(Input(0.5), 1),
                    TrainingDatum(repeated_input1, 1),
                ),
            }
        )

        mlgp = MultiLevelGaussianProcess([fakes.WhiteNoiseGP() for _ in training_data1])
        mlgp.fit(training_data1)

        with self.assertRaisesRegex(
            ValueError,
            exact(
                "Training inputs across all levels must be distinct, but found "
                f"common input {repeated_input1} at levels {problem_level1}, {problem_level2}."
            ),
        ):
            self.compute_multi_level_loo_samples(
                mlgp, domain, costs=MultiLevel([1, 1, 1])
            )

    def test_number_of_design_points_returned_equals_batch_size(self):
        """Then number of design points returned equals the provided batch size."""

        for batch_size in [1, 2, 3]:
            with self.subTest(batch_size=batch_size):
                design_points = self.compute_multi_level_loo_samples(
                    batch_size=batch_size
                )
                self.assertEqual(batch_size, len(design_points.get(1, (1,))))

    def test_returns_design_points_from_domain(self):
        """The return type is a tuple containing pair ``(level, Input)``, with ``level``
        being one of the levels from the supplied multi-level GP and
        each ``Input`` being an input belonging to the supplied simulator domain."""

        domains = [SimulatorDomain([(0, 1)]), SimulatorDomain([(2, 3)])]
        gp1 = MogpEmulator()
        gp1.fit(
            [
                TrainingDatum(Input(0.2), 1),
                TrainingDatum(Input(0.4), 2),
                TrainingDatum(Input(0.6), 3),
            ]
        )
        gp2 = MogpEmulator()
        gp2.fit(
            [
                TrainingDatum(Input(2.2), -1),
                TrainingDatum(Input(2.4), -2),
                TrainingDatum(Input(2.6), -3),
            ]
        )
        costs = self.make_level_costs([1])
        gps = [gp1, gp2]
        for domain, gp in zip(domains, gps):
            with self.subTest(domain=domain, gp=gp):
                mlgp = MultiLevelGaussianProcess([gp])
                design_points = self.compute_multi_level_loo_samples(
                    mlgp=mlgp, domain=domain, costs=costs, batch_size=2
                )

                self.assertIsInstance(design_points, MultiLevel)
                for _, values in design_points.items():
                    for input in values:
                        self.assertIsInstance(input, Input)
                        self.assertIn(input, domain)

    def test_single_batch_level_that_maximises_pei(self):
        """For a single batch output, the input and level returned are the ones that
        maximise weighted pseudo-expected improvements for the leave-one-out error GPs
        across all simulator levels. The weightings are reciprocals of the associated
        costs for calculating differences of simulator outputs."""

        costs = self.make_level_costs([0.0001, 11, 110])
        domain = SimulatorDomain([(0, 1)])
        mlgp = MultiLevelGaussianProcess([MogpEmulator(), MogpEmulator(), MogpEmulator()])
        training_data = MultiLevel(
            {
                1: [
                    TrainingDatum(Input(0.1), 1),
                    TrainingDatum(Input(0.2), 2),
                    TrainingDatum(Input(0.3), 3),
                ],
                2: [
                    TrainingDatum(Input(0.4), 2),
                    TrainingDatum(Input(0.5), 99),
                    TrainingDatum(Input(0.6), -4),
                ],
                3: [
                    TrainingDatum(Input(0.7), 3),
                    TrainingDatum(Input(0.8), -3),
                    TrainingDatum(Input(0.9), 3),
                ],
            }
        )
        mlgp.fit(training_data)

        design_points_ml = self.compute_multi_level_loo_samples(
            mlgp=mlgp, domain=domain, costs=costs
        )

        self.assertEqual(1, len(design_points_ml))

        ml_errors_gp = compute_multi_level_loo_errors_gp(mlgp, domain)
        ml_pei = ml_errors_gp.map(lambda _, gp: PEICalculator(domain, gp))
        _, max_pei1 = maximise(lambda x: ml_pei[1].compute(x), domain)
        _, max_pei2 = maximise(lambda x: ml_pei[2].compute(x), domain)
        _, max_pei3 = maximise(lambda x: ml_pei[3].compute(x), domain)

        expected_level, _ = max(
            [
                (1, max_pei1 / costs[1]),
                (2, max_pei2 / costs[2]),
                (3, max_pei3 / costs[3]),
            ],
            key=lambda tup: tup[1],
        )

        level = list(design_points_ml.keys())[0]
        self.assertEqual(expected_level, level)

    def test_new_design_points_in_batch_distinct(self):
        """A batch of new design points consists of Input objects that are (likely)
        all distinct."""

        design_pts_ml = self.compute_multi_level_loo_samples(batch_size=2).get(1, (1, 2))

        self.assertFalse(
            equal_within_tolerance(
                design_pts_ml[0],
                design_pts_ml[1],
                rel_tol=self.tolerance,
                abs_tol=self.tolerance,
            )
        )

    def test_new_design_points_distinct_from_training_inputs(self):
        """A batch of new design points consists of Input objects that are (likely)
        distinct from the training data inputs across the different levels."""

        design_pts_ml = self.compute_multi_level_loo_samples(
            mlgp=self.default_mlgp, batch_size=3
        )
        training_inputs = []

        for level in design_pts_ml.keys():
            training_inputs.extend(
                datum.input for datum in self.default_mlgp[level].training_data
            )

        for training_x, x in itertools.product(
            training_inputs, list(design_pts_ml.values())
        ):
            with self.subTest(training_input=training_x, design_pt=x):
                self.assertFalse(
                    equal_within_tolerance(
                        training_x,
                        x,
                        rel_tol=self.tolerance,
                        abs_tol=self.tolerance,
                    )
                )

    def test_additional_repulsion_pts_used_for_multi_level(self):
        """If additional repulsion points are provided, then these are used in the
        calculation of pseudo-expected improvement for the LOO errors GP across all levels.
        """

        costs = self.make_level_costs([1, 11, 110])
        domain = SimulatorDomain([(0, 1)])
        mlgp = MultiLevelGaussianProcess([MogpEmulator(), MogpEmulator(), MogpEmulator()])
        training_data = MultiLevel(
            {
                1: [
                    TrainingDatum(Input(0.1), 1),
                    TrainingDatum(Input(0.2), 2),
                    TrainingDatum(Input(0.3), 3),
                ],
                2: [
                    TrainingDatum(Input(0.4), 2),
                    TrainingDatum(Input(0.5), 99),
                    TrainingDatum(Input(0.6), -4),
                ],
                3: [
                    TrainingDatum(Input(0.7), 3),
                    TrainingDatum(Input(0.8), -3),
                    TrainingDatum(Input(0.9), 3),
                ],
            }
        )

        mlgp.fit(training_data)

        # Compute a new design point
        design_pt_ml = compute_multi_level_loo_samples(mlgp, domain, costs)
        design_pt = list(design_pt_ml.values())[0]

        # This creates a MultiLevel for the values to equal None which are not the level for the design point
        repulsion_pts = MultiLevel(
            {
                lvl: (design_pt if lvl in design_pt_ml.keys() else None)
                for lvl in mlgp.levels
            }
        )

        # Re-run computation but now using the new design point as a repulsion point.
        # Should find different design points created.
        design_pt2 = list(
            compute_multi_level_loo_samples(
                mlgp,
                domain,
                costs,
                additional_repulsion_pts=repulsion_pts,
            ).values()
        )[0]

        self.assertNotEqualWithinTolerance(
            design_pt2, design_pt, rel_tol=self.tolerance, abs_tol=self.tolerance
        )

    def test_multiple_levels_returned(self):
        """Ensure that when given a large enough batch size, design points are created on
        multiple levels and not just to 1 level."""

        costs = self.make_level_costs([1, 10, 100])
        domain = SimulatorDomain([(0, 1)])
        batch_size = 10
        mlgp = MultiLevelGaussianProcess([MogpEmulator(), MogpEmulator(), MogpEmulator()])
        training_data = MultiLevel(
            {
                1: [
                    TrainingDatum(Input(0.1), 1),
                    TrainingDatum(Input(0.2), 2),
                    TrainingDatum(Input(0.3), 3),
                ],
                2: [
                    TrainingDatum(Input(0.4), 2),
                    TrainingDatum(Input(0.5), 99),
                    TrainingDatum(Input(0.6), -4),
                ],
                3: [
                    TrainingDatum(Input(0.7), 3),
                    TrainingDatum(Input(0.8), -3),
                    TrainingDatum(Input(0.9), 3),
                ],
            }
        )

        mlgp.fit(training_data)

        design_pts_ml = compute_multi_level_loo_samples(mlgp, domain, costs, batch_size)

        self.assertTrue(len(design_pts_ml.keys()) > 1)

    def test_use_of_seed_across_batch(self):
        """Ensure that, if seeds are provided, a different seed is being used to create every
        pseudo-expected improvement design point within a single batch.

        From test_unseeded_by_default (test_optimisation.py), maximise should give slightly
        different results with unseeded but the same args.

        Hence, if seed is correctly used for every design point creation within a batch
        there should be no unique batches across multiple runs."""

        mock_maximise_return = (self.default_domain.scale([0.5]), 1)
        seeds = MultiLevel([99, None])
        batch_size = 5
        with unittest.mock.patch(
            "exauq.core.designers.maximise",
            autospec=True,
            return_value=mock_maximise_return,
        ) as _:
            results = [
                self.compute_multi_level_loo_samples(batch_size=batch_size, seeds=seeds)
                for _ in range(10)
            ]

        for result in results:
            self.assertTupleEqual(list(results[0].values())[0], list(result.values())[0])


# TODO: test that repulsion points are updated with previously calculated inputs in
# batch mode.


class TestCreateDataForMultiLevelLooSampling(ExauqTestCase):
    def setUp(self) -> None:
        self.data = MultiLevel(
            {
                1: [
                    TrainingDatum(Input(0.1), 1),
                    TrainingDatum(Input(0.2), -2),
                    TrainingDatum(Input(0.3), 3),
                ],
                2: [
                    TrainingDatum(Input(0.1), 1.1),
                    TrainingDatum(Input(0.2), -2.2),
                    TrainingDatum(Input(0.3), 3.3),
                ],
                3: [
                    TrainingDatum(Input(0.1), 1.11),
                    TrainingDatum(Input(0.2), -2.22),
                    TrainingDatum(Input(0.3), 3.33),
                ],
            }
        )

    def test_argument_type_errors(self):
        """Ensure that correct TypeErrors are raised for incorrect arguments."""

        arg1 = "test"

        with self.assertRaisesRegex(
            TypeError,
            exact(
                "Expected 'data' to be of type MultiLevel Sequence of TrainingDatum, "
                f"but received {type(arg1)} instead."
            ),
        ):
            _ = create_data_for_multi_level_loo_sampling(arg1)

        with self.assertRaisesRegex(
            TypeError,
            exact(
                "Expected 'correlations' to be of type MultiLevel Real or Real, "
                f"but received {type(arg1)} instead."
            ),
        ):
            _ = create_data_for_multi_level_loo_sampling(self.data, arg1)

    def test_empty_multi_level_data(self):
        """Ensure a correct Warning is raised for a completely empty MultiLevel."""

        test_data = MultiLevel([])

        with self.assertWarnsRegex(
            UserWarning,
            exact(
                "'data' passed was empty and therefore no transformations taken place."
            ),
        ):
            _ = create_data_for_multi_level_loo_sampling(test_data)

    def test_incorrect_length_of_correlations(self):
        """Ensure that a ValueError is raised if an incorrect number of correlations are passed.

        Given the test_data is maxed at level 4, therefore there should be at least 4 levels of correlations.
        It doesn't matter if there are more as this makes it easier for the user to just store all of the correlations
        for their entire mlgp and pass this when using mlas with only a couple of new points on different levels. However,
        there need to be at least 4 levels in order to correlate previous level points from a top-down point of view.
        """

        test_data = MultiLevel(
            {
                3: [TrainingDatum(Input(0.3), 33)],
                4: [TrainingDatum(Input(0.4), 42)],
            }
        )
        correlations = MultiLevel([1, 1])

        with self.assertRaisesRegex(
            ValueError,
            exact(
                f"'Correlations' MultiLevel expected to be provided for at least max level of 'data' - 1: {max(test_data.levels) - 1}, but "
                f"is only of length: {max(correlations.levels)}."
            ),
        ):
            _ = create_data_for_multi_level_loo_sampling(test_data, correlations)

    def test_data_with_only_one_level(self):
        """Ensure the raw values are simply returned if only 1 level of data is passed."""

        test_data = MultiLevel(
            {
                3: [
                    TrainingDatum(Input(0.1), 1),
                    TrainingDatum(Input(0.2), 2),
                    TrainingDatum(Input(0.3), 3),
                ]
            }
        )

        expected_return = test_data
        returned_data = create_data_for_multi_level_loo_sampling(test_data)
        self.assertEqual(expected_return, returned_data)

    def test_no_training_datum_in_MultiLevel(self):
        """Ensure MultiLevels with empty sequences pass through without error."""

        test_data = MultiLevel(
            {
                1: [],
                2: [],
                3: [],
            }
        )

        expected_return = test_data
        returned_data = create_data_for_multi_level_loo_sampling(test_data)
        self.assertEqual(expected_return, returned_data)

    def test_no_level_1_training_data(self):
        """Ensure that the correct functionality occurs without level 1 in training data."""

        outputs = {2: -1, 3: 1, 4: 2}
        data = MultiLevel(
            {
                2: [TrainingDatum(Input(0.1), outputs[2])],
                3: [TrainingDatum(Input(0.1), outputs[3])],
                4: [TrainingDatum(Input(0.1), outputs[4])],
            }
        )
        correlations = MultiLevel([0.2, 0.3, 0.4])

        delta_data = create_data_for_multi_level_loo_sampling(data, correlations)

        expected = MultiLevel(
            {
                2: [],
                3: [],
                4: [TrainingDatum(Input(0.1), outputs[4] - correlations[3] * outputs[3])],
            }
        )

        self.assertEqual(expected, delta_data)

    def test_missing_levels_multi_level_data(self):
        """Ensure that the correct functionality occurs without all levels in training data."""

        outputs = {1: -1, 3: 1, 4: 2}
        data = MultiLevel(
            {
                1: [TrainingDatum(Input(0.1), outputs[1])],
                3: [TrainingDatum(Input(0.2), outputs[3])],
                4: [TrainingDatum(Input(0.2), outputs[4])],
            }
        )
        correlations = MultiLevel([0.2, 0.3, 0.4])

        delta_data = create_data_for_multi_level_loo_sampling(data, correlations)

        expected = MultiLevel(
            {
                1: [TrainingDatum(Input(0.1), outputs[1])],
                3: [],
                4: [TrainingDatum(Input(0.2), outputs[4] - correlations[3] * outputs[3])],
            }
        )

        self.assertEqual(expected, delta_data)

    def test_different_correlations_data_multi_levels(self):
        """Ensure that full MultiLevel correlations can be passed even with missing levels in training data."""

        # Correlations suggests 5 levels in the full mlgp
        corrs = MultiLevel([0.1, 0.2, 0.3, 0.4])

        outputs = {2: -1, 3: 1, 4: 2}
        data = MultiLevel(
            {
                2: [TrainingDatum(Input(0.1), outputs[2])],
                3: [TrainingDatum(Input(0.1), outputs[3])],
                4: [TrainingDatum(Input(0.1), outputs[4])],
            }
        )

        delta_data = create_data_for_multi_level_loo_sampling(data, corrs)

        expected_data = MultiLevel(
            {
                2: [],
                3: [],
                4: [TrainingDatum(Input(0.1), outputs[4] - corrs[3] * outputs[3])],
            }
        )

        # Ensure data is correct
        self.assertEqual(expected_data, delta_data)

    def test_delta_data_contains_inter_level_differences_of_outputs(self):
        """The data returned consists of differences of outputs between successive levels,
        accounting for the correlation between levels. The data returned for the bottom
        level is just the same data supplied at this level."""

        outputs = {1: 10, 2: -1, 3: 1, 4: 2}
        data = MultiLevel(
            {
                1: [TrainingDatum(Input(0.1), outputs[1])],
                2: [TrainingDatum(Input(0.1), outputs[2])],
                3: [TrainingDatum(Input(0.1), outputs[3])],
                4: [TrainingDatum(Input(0.1), outputs[4])],
            }
        )
        correlations = MultiLevel([0.1, 0.2, 0.3])

        delta_data = create_data_for_multi_level_loo_sampling(data, correlations)

        expected = MultiLevel(
            {
                1: [],
                2: [],
                3: [],
                4: [TrainingDatum(Input(0.1), outputs[4] - correlations[3] * outputs[3])],
            }
        )

        self.assertEqual(expected, delta_data)

    def test_delta_data_more_than_one_datum_at_levels(self):
        """In the case where there are multiple training data at some level, the data
        returned consists of differences of outputs between successive levels, accounting
        for the correlation between levels. The data returned for the bottom level is just
        the same data supplied at this level."""

        data = MultiLevel(
            {
                1: [
                    TrainingDatum(Input(0.1), 1),
                    TrainingDatum(Input(0.2), 2),
                    TrainingDatum(Input(0.3), 3),
                    TrainingDatum(Input(0.4), 4),
                ],
                2: [
                    TrainingDatum(Input(0.1), 1.1),
                    TrainingDatum(Input(0.2), 2.2),
                    TrainingDatum(Input(0.3), 3.3),
                ],
                3: [
                    TrainingDatum(Input(0.1), 1.11),
                    TrainingDatum(Input(0.2), 2.22),
                ],
            }
        )
        correlations = MultiLevel([0.1, 0.2])

        delta_data = create_data_for_multi_level_loo_sampling(data, correlations)

        expected = MultiLevel(
            {
                1: [
                    TrainingDatum(Input(0.4), 4),
                ],
                2: [
                    TrainingDatum(Input(0.3), 3.3 - correlations[1] * 3),
                ],
                3: [
                    TrainingDatum(Input(0.1), 1.11 - correlations[2] * 1.1),
                    TrainingDatum(Input(0.2), 2.22 - correlations[2] * 2.2),
                ],
            }
        )

        self.assertEqual(expected, delta_data)

    def test_missing_common_points(self):
        """In the case where there are points at a higher level without a match at the previous
        level, these points should be ignored, with all other deltas returned as usual."""

        data = MultiLevel(
            {
                1: [
                    TrainingDatum(Input(0.1), 1),
                    TrainingDatum(Input(0.2), 2),
                    TrainingDatum(Input(0.3), 3),
                    TrainingDatum(Input(0.4), 4),
                ],
                2: [
                    TrainingDatum(Input(0.1), 1.1),
                    TrainingDatum(Input(0.2), 2.2),
                    TrainingDatum(Input(0.3), 3.3),
                ],
                3: [
                    TrainingDatum(Input(0.1), 1.11),
                    TrainingDatum(Input(0.5), 2.22),
                ],
            }
        )
        correlations = MultiLevel([0.1, 0.2])

        delta_data = create_data_for_multi_level_loo_sampling(data, correlations)

        expected = MultiLevel(
            {
                1: [
                    TrainingDatum(Input(0.4), 4),
                ],
                2: [
                    TrainingDatum(Input(0.2), 2.2 - correlations[1] * 2),
                    TrainingDatum(Input(0.3), 3.3 - correlations[1] * 3),
                ],
                3: [
                    TrainingDatum(Input(0.1), 1.11 - correlations[2] * 1.1),
                ],
            }
        )

        self.assertEqual(expected, delta_data)

    def test_empty_deltas_higher_levels(self):
        """If there are no matching points at the previous level, no deltas should be calculated and
        the higher levels should be returned as empty."""

        data = MultiLevel(
            {
                1: [
                    TrainingDatum(Input(0.1), 1),
                    TrainingDatum(Input(0.2), 2),
                    TrainingDatum(Input(0.3), 3),
                    TrainingDatum(Input(0.4), 4),
                ],
                2: [
                    TrainingDatum(Input(0.11), 1.1),
                    TrainingDatum(Input(0.22), 2.2),
                    TrainingDatum(Input(0.33), 3.3),
                ],
                3: [
                    TrainingDatum(Input(0.111), 1.11),
                ],
            }
        )
        correlations = MultiLevel([0.1, 0.2])

        delta_data = create_data_for_multi_level_loo_sampling(data, correlations)

        expected = MultiLevel(
            {
                1: [
                    TrainingDatum(Input(0.1), 1),
                    TrainingDatum(Input(0.2), 2),
                    TrainingDatum(Input(0.3), 3),
                    TrainingDatum(Input(0.4), 4),
                ],
                2: [],
                3: [],
            }
        )

        self.assertEqual(expected, delta_data)

    def test_empty_deltas_first_level(self):
        """If all points at the 1st level match those at the 2nd level, the resulting 1st level should be empty."""

        data = MultiLevel(
            {
                1: [
                    TrainingDatum(Input(0.1), 1),
                    TrainingDatum(Input(0.2), 2),
                    TrainingDatum(Input(0.3), 3),
                    TrainingDatum(Input(0.4), 4),
                ],
                2: [
                    TrainingDatum(Input(0.1), 1.1),
                    TrainingDatum(Input(0.2), 2.2),
                    TrainingDatum(Input(0.3), 3.3),
                    TrainingDatum(Input(0.4), 4.4),
                ],
            }
        )
        correlations = MultiLevel([0.1])

        delta_data = create_data_for_multi_level_loo_sampling(data, correlations)

        expected = MultiLevel(
            {
                1: [],
                2: [
                    TrainingDatum(Input(0.1), 1.1 - correlations[1] * 1),
                    TrainingDatum(Input(0.2), 2.2 - correlations[1] * 2),
                    TrainingDatum(Input(0.3), 3.3 - correlations[1] * 3),
                    TrainingDatum(Input(0.4), 4.4 - correlations[1] * 4),
                ],
            }
        )

        self.assertEqual(expected, delta_data)

    def test_empty_deltas_warning(self):
        """Ensure a correct Warning is raised if a resulting level ends up empty."""

        test_data = MultiLevel(
            {
                1: [
                    TrainingDatum(Input(0.1), 1),
                    TrainingDatum(Input(0.2), 2),
                    TrainingDatum(Input(0.3), 3),
                    TrainingDatum(Input(0.4), 4),
                ],
                2: [],
                3: [],
            }
        )
        with catch_warnings(record=True) as w:
            simplefilter("always")

            _ = create_data_for_multi_level_loo_sampling(self.data)

            # Check that two warnings were raised
            self.assertEqual(len(w), 2)

            # Check warning messages
            self.assertEqual(
                str(w[0].message),
                "After processing, Level 1 is empty. Check your input data",
            )
            self.assertEqual(
                str(w[1].message),
                "After processing, Level 2 is empty. Check your input data",
            )

            # Check warning types
            self.assertTrue(
                all(issubclass(warning.category, UserWarning) for warning in w)
            )

        with catch_warnings(record=True) as w:
            simplefilter("always")

            _ = create_data_for_multi_level_loo_sampling(test_data)

            # Check that two warnings were raised
            self.assertEqual(len(w), 2)

            # Check warning messages
            self.assertEqual(
                str(w[0].message),
                "After processing, Level 2 is empty. Check your input data",
            )
            self.assertEqual(
                str(w[1].message),
                "After processing, Level 3 is empty. Check your input data",
            )

            # Check warning types
            self.assertTrue(
                all(issubclass(warning.category, UserWarning) for warning in w)
            )


class TestComputeDeltaCoefficients(ExauqTestCase):

    def setUp(self) -> None:

        self.levels = range(1, 5)

    def test_argument_type_errors(self):
        """Ensure that the correct TypeErrors are raised for arguments passed."""

        arg1 = 0.6

        with self.assertRaisesRegex(
            TypeError,
            exact(
                "Expected 'levels' to be of type Sequence of int or int, "
                f"but received {type(arg1)} instead."
            ),
        ):
            _ = compute_delta_coefficients(arg1)

        arg2 = "test"

        with self.assertRaisesRegex(
            TypeError,
            exact(
                "Expected 'correlations' to be of type MultiLevel Real or Real, "
                f"but received {type(arg2)} instead."
            ),
        ):
            _ = compute_delta_coefficients(self.levels, arg2)

        arg3 = [1, 2, 2.9]

        with self.assertRaisesRegex(
            TypeError,
            exact(
                "Expected 'levels' to be of type Sequence of int or int, "
                "but received unexpected types."
            ),
        ):
            _ = compute_delta_coefficients(arg3)

    def test_default_correlation(self):
        """By default, a constant correlation of 1 is applied to every level
        (except the top level)."""

        delta_coefficients = compute_delta_coefficients(self.levels)

        expected = MultiLevel({level: 1 for level in self.levels})
        self.assertEqual(expected, delta_coefficients)

    def test_delta_coefficients_constant_correlation_case(self):
        """If a single real number is supplied for the correlations, then this is
        interpreted as applying to every level (except the top level)."""

        correlation = 0.5

        delta_coefficients = compute_delta_coefficients(
            self.levels, correlations=correlation
        )

        expected = MultiLevel(
            {1: correlation**3, 2: correlation**2, 3: correlation, 4: 1}
        )
        self.assertEqual(expected, delta_coefficients)

    def test_returns_delta_coefficients(self):
        """The coefficients for creating the multi-level GP for LOO adaptive sampling
        are returned, where the highest level coefficient is equal to 1 and each of the
        lower coefficients are products of the supplied correlation values."""

        data = MultiLevel(
            {
                1: [
                    TrainingDatum(Input(0.1), 1),
                ],
                2: [
                    TrainingDatum(Input(0.1), 1),
                ],
                3: [
                    TrainingDatum(Input(0.1), 1),
                ],
                4: [
                    TrainingDatum(Input(0.1), 1),
                ],
            }
        )

        # Case of 2 levels
        corrs2 = MultiLevel([0.1])

        delta_coefficients = compute_delta_coefficients(2, corrs2)

        expected = MultiLevel({1: corrs2[1], 2: 1})
        self.assertEqual(expected, delta_coefficients)

        # Case of 3 levels
        corrs3 = MultiLevel([0.1, 0.2])

        delta_coefficients = compute_delta_coefficients(3, corrs3)

        expected = MultiLevel({1: corrs3[1] * corrs3[2], 2: corrs3[2], 3: 1})
        self.assertEqual(expected, delta_coefficients)

        # Case of 4 levels
        corrs4 = MultiLevel([0.1, 0.2, 0.3])

        delta_coefficients = compute_delta_coefficients(4, corrs4)

        expected = MultiLevel(
            {
                1: corrs4[1] * corrs4[2] * corrs4[3],
                2: corrs4[2] * corrs4[3],
                3: corrs4[3],
                4: 1,
            }
        )
        self.assertEqual(expected, delta_coefficients)


class TestRemoveMultiLevelRepeatedInput(ExauqTestCase):

    def test_multi_level_removal_repeated_inputs(self):
        """Ensure that repeating inputs across levels are removed."""

        data = MultiLevel(
            {
                1: [TrainingDatum(Input(0.3), 3)],
                2: [TrainingDatum(Input(0.3), 3.3)],
                3: [TrainingDatum(Input(0.3), 3.33)],
            }
        )

        expected_return = MultiLevel({1: [], 2: [], 3: [TrainingDatum(Input(0.3), 3.33)]})

        level = 3
        datum = data[level][0]

        training_data = _remove_multi_level_repeated_input(data, datum, level)

        self.assertEqual(training_data, expected_return)

    def test_multi_level_empty_level_data(self):
        """Ensure that there are no errors with empty levels"""

        data = MultiLevel(
            {
                1: [TrainingDatum(Input(0.3), 3)],
                2: [],
                3: [TrainingDatum(Input(0.3), 3.33)],
            }
        )

        expected_return = MultiLevel({1: [], 2: [], 3: [TrainingDatum(Input(0.3), 3.33)]})

        level = 3
        datum = data[level][0]

        training_data = _remove_multi_level_repeated_input(data, datum, level)

        self.assertEqual(training_data, expected_return)

    def test_multi_level_repeated_inputs_within_levels(self):
        """Ensure if there are repeated inputs within lower levels, these are all removed."""

        data = MultiLevel(
            {
                1: [TrainingDatum(Input(0.3), 3)],
                2: [TrainingDatum(Input(0.3), 3), TrainingDatum(Input(0.3), 3)],
                3: [TrainingDatum(Input(0.3), 3.33)],
            }
        )

        expected_return = MultiLevel({1: [], 2: [], 3: [TrainingDatum(Input(0.3), 3.33)]})

        level = 3
        datum = data[level][0]

        training_data = _remove_multi_level_repeated_input(data, datum, level)

        self.assertEqual(training_data, expected_return)

    def test_multi_level_repeated_inputs_in_highest_level(self):
        """Ensure that if there are repeated inputs within the highest level, these
        are all removed."""

        data = MultiLevel(
            {
                1: [TrainingDatum(Input(0.3), 3)],
                2: [TrainingDatum(Input(0.3), 3)],
                3: [TrainingDatum(Input(0.3), 3.33), TrainingDatum(Input(0.3), 3.33)],
            }
        )

        expected_return = MultiLevel({1: [], 2: [], 3: [TrainingDatum(Input(0.3), 3.33)]})

        level = 3
        datum = data[level][0]

        training_data = _remove_multi_level_repeated_input(data, datum, level)

        self.assertEqual(training_data, expected_return)

    def test_multi_level_no_repeated_inputs(self):
        """Ensure that if there are no repeated inputs then the entire Multilevel
        is returned untouched."""

        data = MultiLevel(
            {
                1: [TrainingDatum(Input(0.1), 1)],
                2: [TrainingDatum(Input(0.2), 2.2)],
                3: [TrainingDatum(Input(0.3), 3.33)],
            }
        )

        expected_return = data
        level = 3
        datum = data[level][0]

        training_data = _remove_multi_level_repeated_input(data, datum, level)

        self.assertEqual(training_data, expected_return)

    def test_multi_level_missing_level(self):
        """Ensure that with missing levels, the correct repeating inputs are still removed"""

        data = MultiLevel(
            {1: [TrainingDatum(Input(0.3), 3)], 3: [TrainingDatum(Input(0.3), 3.33)]}
        )

        expected_return = MultiLevel({1: [], 3: [TrainingDatum(Input(0.3), 3.33)]})

        level = 3
        datum = data[level][0]

        training_data = _remove_multi_level_repeated_input(data, datum, level)

        self.assertEqual(training_data, expected_return)


if __name__ == "__main__":

    unittest.main()
