import itertools
import math
import unittest
import unittest.mock

import mogp_emulator as mogp
import numpy as np

from exauq.core.emulators import MogpEmulator, MogpHyperparameters
from exauq.core.modelling import (
    GaussianProcessHyperparameters,
    Input,
    Prediction,
    TrainingDatum,
)
from tests.utilities.utilities import ExauqTestCase, exact


class TestMogpEmulator(ExauqTestCase):
    def setUp(self) -> None:
        # Some default args to use for constructing mogp GaussianProcess objects
        self.inputs_arr = np.array(
            [[0, 0], [0.2, 0.1], [0.3, 0.5], [0.7, 0.4], [0.9, 0.8]]
        )
        self.targets_arr = np.array([1, 2, 3.1, 9, 2])

        self.inputs3 = [Input(1, 1), Input(2, 2), Input(3, 3)]
        self.inputs4 = [Input(10, 10), Input(20, 20)]

        # kwargs for constructing an mogp GuassianProcess object
        self.gp_kwargs = {
            "mean": None,
            "kernel": "Matern52",  # non-default
            "priors": None,
            "nugget": "pivot",  # non-default
            "inputdict": {},
            "use_patsy": False,  # non-default
        }

        # Default training data for fitting an emulator
        self.training_data = [
            TrainingDatum(Input(0, 0), 1),
            TrainingDatum(Input(0.2, 0.1), 2),
            TrainingDatum(Input(0.3, 0.5), 3),
            TrainingDatum(Input(0.7, 0.4), 4),
            TrainingDatum(Input(0.9, 0.8), 5),
        ]

        # Input not contained in training data, for making predictions
        self.x = Input(0.1, 0.1)

    def assertAlmostBetween(self, x, lower, upper, **kwargs) -> None:
        """Checks whether a number lies between bounds up to a tolerance.

        An AssertionError is thrown unless `lower` < `x` < `upper` or `x` is
        close to either bound. Keyword arguments are passed to the function
        `math.isclose`, with `rel_tol` and `abs_tol` being used to define the
        tolerances in the closeness check (see the documentation for
        `math.isclose` on their definition and default values).
        """
        almost_between = (
            (lower < x < upper)
            or math.isclose(x, lower, **kwargs)
            or math.isclose(x, upper, **kwargs)
        )
        assert almost_between, "'x' not between lower and upper bounds to given tolerance"

    def test_initialiser(self):
        """Test that an instance of MogpEmulator can be initialised from args
        and kwargs required of an mogp GaussianProcess object.
        """

        MogpEmulator(**self.gp_kwargs)

    def test_initialiser_error(self):
        """Test that a RuntimeError is raised if a mogp GaussianProcess
        couldn't be initialised.
        """

        with self.assertRaises(RuntimeError) as cm:
            MogpEmulator(men=None)

        expected_msg = (
            "Could not construct mogp-emulator GaussianProcess "
            "during initialisation of MogpEmulator"
        )
        self.assertEqual(expected_msg, str(cm.exception))

    @unittest.mock.patch("exauq.core.emulators.GaussianProcess")
    def test_initialiser_base_exception_bubbled(self, mock):
        """Test that an exception that isn't derived from Exception gets raised as-is."""

        mock.side_effect = BaseException("msg")
        with self.assertRaises(BaseException) as cm:
            MogpEmulator()

        self.assertEqual("msg", str(cm.exception))

    def test_underlying_gp_kwargs(self):
        """Test that the underlying mogp GaussianProcess was constructed with
        kwargs supplied to the MogpEmulator initialiser."""

        # Non-default choices for some kwargs
        emulator = MogpEmulator(
            kernel=self.gp_kwargs["kernel"], nugget=self.gp_kwargs["nugget"]
        )

        self.assertEqual("Matern 5/2 Kernel", str(emulator.gp.kernel))
        self.assertEqual(self.gp_kwargs["nugget"], emulator.gp.nugget_type)

    def test_initialiser_inputs_targets_ignored(self):
        """Test that inputs and targets kwargs are ignored when initialising
        an emulator."""

        emulator = MogpEmulator(
            inputs=np.array([[0, 0], [0.2, 0.1]]), targets=np.array([1, 2])
        )
        self.assertEqual(0, emulator.gp.inputs.size)
        self.assertEqual(0, emulator.gp.targets.size)

    def test_initialiser_invalid_kernel_error(self):
        """A ValueError is raised if the supplied kernel does not describe one of the
        supported kernel functions."""

        kernel = "UniformSqExp"
        with self.assertRaisesRegex(
            ValueError,
            exact(
                f"Could not initialise MogpEmulator with kernel = {kernel}: not a "
                "supported kernel function."
            ),
        ):
            _ = MogpEmulator(kernel=kernel)

    def test_training_data(self):
        """Test that the training data in the underlying GP and the
        training_data property is empty when an emulator is constructed."""

        emulator = MogpEmulator()
        self.assertEqual(0, emulator.gp.inputs.size)
        self.assertEqual(0, emulator.gp.targets.size)
        self.assertEqual(tuple(), emulator.training_data)

    def test_correlation_arg_validation(self):
        """A TypeError is raised if one of the args is not a sequence of Input objects."""

        params = MogpHyperparameters(corr_length_scales=[2], process_var=1, nugget=1)
        emulator = MogpEmulator()
        training_data = [
            TrainingDatum(Input(0), 1),
            TrainingDatum(Input(0.2), 1),
            TrainingDatum(Input(0.4), 1),
            TrainingDatum(Input(0.6), 1),
        ]
        emulator.fit(training_data, hyperparameters=params)

        inputs1 = 1
        inputs2 = [Input(3)]
        with self.assertRaisesRegex(
            TypeError,
            exact(
                "Expected 'inputs1' and 'inputs2' to be sequences of Input objects, but received "
                f"{type(inputs1)} and {type(inputs2)} instead."
            ),
        ):
            _ = emulator.correlation(inputs1, inputs2)

        inputs1 = Input(1)
        inputs2 = [Input(3)]
        with self.assertRaisesRegex(
            TypeError,
            exact("Expected 'inputs1' and 'inputs2' to only contain Input objects."),
        ):
            _ = emulator.correlation(inputs1, inputs2)

        inputs1 = [Input(1)]
        inputs2 = Input(3)
        with self.assertRaisesRegex(
            TypeError,
            exact("Expected 'inputs1' and 'inputs2' to only contain Input objects."),
        ):
            _ = emulator.correlation(inputs1, inputs2)

    def test_correlation_not_trained_error(self):
        """An AssertionError is raised if the GP has not been trained on data."""

        emulator = MogpEmulator()
        x1, x2 = self.inputs3[0], self.inputs4[0]
        with self.assertRaisesRegex(
            AssertionError,
            exact(
                f"Cannot calculate correlations for this instance of {emulator.__class__} because "
                "it hasn't yet been trained on data."
            ),
        ):
            _ = emulator.correlation([x1], [x2])

    def test_correlation_empty_input_sequences(self):
        """An empty tuple is returned if either of the supplied input sequences is empty."""

        params = MogpHyperparameters(corr_length_scales=[1, 2], process_var=1, nugget=1)
        emulator = MogpEmulator()
        emulator.fit(self.training_data, hyperparameters=params)

        inputs = [Input(1, 1), Input(2, 4)]
        self.assertEqual(tuple(), emulator.correlation([], inputs))
        self.assertEqual(tuple(), emulator.correlation(inputs, []))

    def test_correlation_incompatible_dimensions_error(self):
        """A ValueError is raised if the dimensions one of the inputs does not agree with
        the number dimension of inputs used to train the GP."""

        params = MogpHyperparameters(corr_length_scales=[1, 2], process_var=1, nugget=1)
        emulator = MogpEmulator()
        emulator.fit(self.training_data, hyperparameters=params)
        expected_dim = len(self.training_data[0].input)

        inputs1, inputs2 = [Input(0), Input(1, 1)], [Input(0, 1)]
        wrong_dim = len(inputs1[0])
        with self.assertRaisesRegex(
            ValueError,
            exact(
                f"Expected inputs to have dimension equal to {expected_dim}, but received input of "
                f"dimension {wrong_dim}."
            ),
        ):
            _ = emulator.correlation(inputs1, inputs2)

        inputs1, inputs2 = [Input(0, 1)], [Input(1, 1), Input(0)]
        wrong_dim = len(inputs2[1])
        with self.assertRaisesRegex(
            ValueError,
            exact(
                f"Expected inputs to have dimension equal to {expected_dim}, but received input of "
                f"dimension {wrong_dim}."
            ),
        ):
            _ = emulator.correlation(inputs1, inputs2)

    def test_correlation_matrix_entries_given_by_kernels(self):
        """The calculation of correlations agrees with the pairwise kernel calculations
        for the kernel chosen for the underlying MOGP GaussianProcess."""

        kernel_funcs = {
            "Matern52": mogp.Kernel.Matern52().kernel_f,
            "SquaredExponential": mogp.Kernel.SquaredExponential().kernel_f,
            "ProductMat52": mogp.Kernel.ProductMat52().kernel_f,
        }

        params = MogpHyperparameters(corr_length_scales=[1, 2], process_var=1, nugget=1)
        corr_raw = params.to_mogp_gp_params().corr_raw

        for kernel in kernel_funcs:
            with self.subTest(kernel=kernel):
                # Train an emulator with some hyperparameters and a specific kernel
                emulator = MogpEmulator(kernel=kernel)
                emulator.fit(self.training_data, hyperparameters=params)

                kernel_f = kernel_funcs[kernel]
                correlations = emulator.correlation(self.inputs3, self.inputs4)
                for (i1, x1), (i2, x2) in zip(
                    enumerate(self.inputs3), enumerate(self.inputs4)
                ):
                    self.assertEqualWithinTolerance(
                        kernel_f(np.array(x1), np.array(x2), corr_raw)[0, 0],
                        correlations[i1][i2],
                    )

    def test_correlation_default_kernel(self):
        """If a kernel is not specified at MogpEmulator initialisation, then a
        squared exponential kernel will be used."""

        params = MogpHyperparameters(corr_length_scales=[1, 2], process_var=1, nugget=1)
        corr_raw = params.to_mogp_gp_params().corr_raw

        emulator = MogpEmulator()
        emulator.fit(self.training_data, hyperparameters=params)

        correlations = emulator.correlation(self.inputs3, self.inputs4)
        for (i1, x1), (i2, x2) in zip(enumerate(self.inputs3), enumerate(self.inputs4)):
            self.assertEqualWithinTolerance(
                mogp.Kernel.SquaredExponential().kernel_f(
                    np.array(x1), np.array(x2), corr_raw
                )[0, 0],
                correlations[i1][i2],
            )

    def test_correlation_depends_on_fit_correlations(self):
        """The correlation calculation depends on the correlation length scales that the
        GP is fit with, but not the process variance or nugget."""

        corr = [1, 2]
        process_var = 1
        nugget = 1
        x1, x2 = self.inputs3[0], self.inputs4[0]
        gp = MogpEmulator()

        # Check correlation differs when correlation length scales changed
        gp.fit(
            self.training_data,
            hyperparameters=MogpHyperparameters(
                corr_length_scales=[1, 2], process_var=process_var, nugget=nugget
            ),
        )
        correlation1 = gp.correlation([x1], [x2])[0][0]
        gp.fit(
            self.training_data,
            hyperparameters=MogpHyperparameters(
                corr_length_scales=[10, 20], process_var=process_var, nugget=nugget
            ),
        )
        correlation2 = gp.correlation([x1], [x2])[0][0]
        self.assertNotEqual(correlation1, correlation2)

        # Check correlation unchanged when only process variance changed
        gp.fit(
            self.training_data,
            hyperparameters=MogpHyperparameters(
                corr_length_scales=corr, process_var=1, nugget=nugget
            ),
        )
        correlation1 = gp.correlation([x1], [x2])[0][0]
        gp.fit(
            self.training_data,
            hyperparameters=MogpHyperparameters(
                corr_length_scales=corr, process_var=2, nugget=nugget
            ),
        )
        correlation2 = gp.correlation([x1], [x2])[0][0]
        self.assertEqual(correlation1, correlation2)

        # Check correlation unchanged when only nugget changed
        gp.fit(
            self.training_data,
            hyperparameters=MogpHyperparameters(
                corr_length_scales=corr, process_var=process_var, nugget=1
            ),
        )
        correlation1 = gp.correlation([x1], [x2])[0][0]
        gp.fit(
            self.training_data,
            hyperparameters=MogpHyperparameters(
                corr_length_scales=corr, process_var=process_var, nugget=2
            ),
        )
        correlation2 = gp.correlation([x1], [x2])[0][0]
        self.assertEqual(correlation1, correlation2)

    def test_covariance_matrix_empty_sequence_arg(self):
        """An empty tuple is returned if the supplied input sequences is empty."""

        params = MogpHyperparameters(corr_length_scales=[1, 2], process_var=1, nugget=1)
        emulator = MogpEmulator()
        emulator.fit(self.training_data, hyperparameters=params)

        self.assertEqual(tuple(), emulator.covariance_matrix([]))

    def test_covariance_matrix_not_trained(self):
        """An empty tuple is returned if the GP has not been trained on data."""

        emulator = MogpEmulator()
        self.assertEqual(tuple(), emulator.covariance_matrix([Input(1)]))

    def test_covariance_matrix_correlations_with_training_data(self):
        """The covariance matrix is equal to the correlation between the supplied inputs
        and the training data inputs."""

        params = MogpHyperparameters(corr_length_scales=[1, 2], process_var=1, nugget=1)
        emulator = MogpEmulator()
        emulator.fit(self.training_data, hyperparameters=params)
        inputs = [Input(0.1, 0.9), Input(0.2, 0.2)]
        self.assertEqual(
            emulator.correlation([d.input for d in self.training_data], inputs),
            emulator.covariance_matrix(inputs),
        )

    def test_covariance_matrix_arg_validation(self):

        params = MogpHyperparameters(corr_length_scales=[1, 2], process_var=1, nugget=1)
        emulator = MogpEmulator()
        emulator.fit(self.training_data, hyperparameters=params)

        inputs = 1
        with self.assertRaisesRegex(
            TypeError,
            exact(
                "Expected 'inputs' to be a sequence of Input objects, but received "
                f"{type(inputs)} instead."
            ),
        ):
            _ = emulator.covariance_matrix(inputs)

        inputs = Input(1)
        with self.assertRaisesRegex(
            TypeError,
            exact("Expected 'inputs' to only contain Input objects."),
        ):
            _ = emulator.covariance_matrix(inputs)

    def test_covariance_matrix_incompatible_dimensions_error(self):
        """A ValueError is raised if the dimensions of one of the inputs does not agree with
        the number dimension of inputs used to train the GP."""

        params = MogpHyperparameters(corr_length_scales=[1, 2], process_var=1, nugget=1)
        emulator = MogpEmulator()
        emulator.fit(self.training_data, hyperparameters=params)
        expected_dim = len(self.training_data[0].input)

        inputs = [Input(1, 1), Input(0)]
        wrong_dim = len(inputs[1])
        with self.assertRaisesRegex(
            ValueError,
            exact(
                f"Expected inputs to have dimension equal to {expected_dim}, but received input of "
                f"dimension {wrong_dim}."
            ),
        ):
            _ = emulator.covariance_matrix(inputs)

    def test_fit_raises_value_error_if_infinite_training_data_supplied(self):
        """A ValueError is raised if one attempts to fit the emulator to an infinite
        collection of training data."""

        emulator = MogpEmulator()

        # Mock a stream of unsizeable data (note it doesn't implement __len__ so
        # doesn't define a collection).
        def unsizeable_data():
            for _ in range(1000):
                yield TrainingDatum(Input(0), 1)

        for data in [1, TrainingDatum(Input(0), 1), unsizeable_data()]:
            with self.subTest(data=data):
                with self.assertRaisesRegex(
                    TypeError,
                    exact(
                        f"Expected a finite collection of TrainingDatum, but received {type(data)}."
                    ),
                ):
                    emulator.fit(data)

    def test_fit_allows_finite_collections_of_training_data(self):
        """The emulator can be fit on (finite) collections of training data."""

        emulator = MogpEmulator()
        training_data = [TrainingDatum(Input(0), 1), TrainingDatum(Input(0.5), 1)]
        for data in [training_data, tuple(training_data)]:
            try:
                _ = emulator.fit(data)
            except Exception:
                self.fail(f"Should not have failed to fit emulator with data = {data}")

    def test_fit_estimates_hyperparameters_by_default(self):
        """Test that fitting the emulator results in the underlying GP being fit
        with hyperparameter estimation."""

        gp = mogp.fit_GP_MAP(mogp.GaussianProcess(self.inputs_arr, self.targets_arr))
        emulator = MogpEmulator()
        emulator.fit(TrainingDatum.list_from_arrays(self.inputs_arr, self.targets_arr))

        # Note: need to use allclose because fitting is not deterministic.
        tolerance = 1e-5
        self.assertEqualWithinTolerance(
            gp.theta.corr,
            emulator.fit_hyperparameters.corr_length_scales,
            rel_tol=tolerance,
            abs_tol=tolerance,
        )
        self.assertEqualWithinTolerance(
            gp.theta.cov,
            emulator.fit_hyperparameters.process_var,
            rel_tol=tolerance,
            abs_tol=tolerance,
        )
        self.assertEqualWithinTolerance(
            gp.theta.nugget,
            emulator.fit_hyperparameters.nugget,
            rel_tol=tolerance,
            abs_tol=tolerance,
        )

    def test_fit_training_data(self):
        """Test that the emulator is trained on the supplied training data
        and its training_data property is updated."""

        emulator = MogpEmulator()
        training_data = tuple(
            TrainingDatum.list_from_arrays(self.inputs_arr, self.targets_arr)
        )
        emulator.fit(training_data)

        self.assertEqualWithinTolerance(self.inputs_arr, emulator.gp.inputs)
        self.assertEqualWithinTolerance(self.targets_arr, emulator.gp.targets)
        self.assertEqual(training_data, emulator.training_data)

    def test_fit_with_bounds_error(self):
        """Test that a ValueError is raised if a non-positive value is supplied for
        one of the upper bounds."""

        emulator = MogpEmulator()
        training_data = [TrainingDatum(Input(0.5), 0.5)]

        with self.assertRaises(ValueError) as cm:
            emulator.fit(training_data, hyperparameter_bounds=[(None, -1)])

        expected_msg = "Upper bounds must be positive numbers"
        self.assertEqual(expected_msg, str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            emulator.fit(training_data, hyperparameter_bounds=[(None, 0)])

        expected_msg = "Upper bounds must be positive numbers"
        self.assertEqual(expected_msg, str(cm.exception))

    def test_fit_with_bounds(self):
        """Test that fitting the emulator respects bounds on hyperparameters
        when these are supplied."""

        gp = mogp.fit_GP_MAP(mogp.GaussianProcess(self.inputs_arr, self.targets_arr))

        # Compute bounds to apply, by creating small windows away from known
        # optimal values of the hyperparameters.
        corr = gp.theta.corr
        cov = gp.theta.cov
        bounds = (
            (0.8 * corr[0], 0.9 * corr[0]),
            (0.8 * corr[1], 0.9 * corr[1]),
            (1.1 * cov, 1.2 * cov),
        )

        emulator = MogpEmulator()
        emulator.fit(
            TrainingDatum.list_from_arrays(self.inputs_arr, self.targets_arr),
            hyperparameter_bounds=bounds,
        )
        actual_corr = emulator.gp.theta.corr
        actual_cov = emulator.gp.theta.cov

        # Note: need to check values are between bounds up to a tolerance
        # due to floating point inprecision.
        self.assertAlmostBetween(
            actual_corr[0], bounds[0][0], bounds[0][1], rel_tol=1e-10
        )
        self.assertAlmostBetween(
            actual_corr[1], bounds[1][0], bounds[1][1], rel_tol=1e-10
        )
        self.assertAlmostBetween(actual_cov, bounds[2][0], bounds[2][1], rel_tol=1e-10)

    def test_fit_with_no_upper_bound(self):
        """Test that if None is supplied as an upper/lower bound then this will be
        interpreted as specifying no upper/lower bound on the corresponding hyperparameter
        in estimation."""

        emulator = MogpEmulator()
        training_data = TrainingDatum.list_from_arrays(self.inputs_arr, self.targets_arr)
        emulator.fit(training_data)
        corr = emulator.fit_hyperparameters.corr_length_scales
        cov = emulator.fit_hyperparameters.process_var

        # Create bounds based on open windows around the 'true' estimates.
        open_bounds = [
            ((0.9 * corr[0], None), (0.9 * corr[1], None), (0.9 * cov, None)),
            ((None, 1.1 * corr[0]), (None, 1.1 * corr[1]), (None, 1.1 * cov)),
            ((None, None), (None, None), (None, None)),
        ]
        for bounds in open_bounds:
            emulator.fit(training_data, hyperparameter_bounds=bounds)

            self.assertEqualWithinTolerance(
                emulator.fit_hyperparameters.corr_length_scales,
                corr,
                rel_tol=1e-5,
                abs_tol=1e-5,
            )
            self.assertEqualWithinTolerance(
                emulator.fit_hyperparameters.process_var, cov, rel_tol=1e-5, abs_tol=1e-5
            )

    def test_fit_gp_kwargs(self):
        """Test that, after fitting, the underlying mogp emulator has the same
        kwarg settings as the original did before fitting."""

        emulator = MogpEmulator(nugget=self.gp_kwargs["nugget"])

        emulator.fit(training_data=self.training_data)

        self.assertEqual(self.gp_kwargs["nugget"], emulator.gp.nugget_type)

    def test_fit_ignores_inputs_targets(self):
        """Test that fitting ignores any inputs and targets supplied during
        initialisation of the emulator."""

        emulator = MogpEmulator(
            inputs=np.array([[0, 0], [0.2, 0.1]]), targets=np.array([1, 2])
        )
        emulator.fit([])

        self.assertEqual(0, emulator.gp.inputs.size)
        self.assertEqual(0, emulator.gp.targets.size)
        self.assertEqual(tuple(), emulator.training_data)

    def test_fit_empty_training_data(self):
        """Test that the training data and underlying mogp GaussianProcess
        are not changed when fitting with no training data."""

        # Case where no data has previously been fit
        emulator = MogpEmulator()

        emulator.fit([])

        self.assertEqual(0, emulator.gp.inputs.size)
        self.assertEqual(0, emulator.gp.targets.size)
        self.assertEqual(tuple(), emulator.training_data)

        # Case where data has previously been fit
        emulator.fit(TrainingDatum.list_from_arrays(self.inputs_arr, self.targets_arr))
        expected_inputs = emulator.gp.inputs
        expected_targets = emulator.gp.targets
        expected_training_data = emulator.training_data

        emulator.fit([])

        self.assertEqualWithinTolerance(expected_inputs, emulator.gp.inputs)
        self.assertEqualWithinTolerance(expected_targets, emulator.gp.targets)
        self.assertEqual(expected_training_data, emulator.training_data)

    def test_fitted_hyperparameters_can_be_retrieved(self):
        """Given an emulator, when it is fit to training data then the fitted
        hyperparameters can be retrieved and agree with those in the underlying
        MOGP GaussianProcess object."""

        emulator = MogpEmulator()
        emulator.fit(self.training_data)
        self.assertEqual(
            MogpHyperparameters.from_mogp_gp_params(emulator.gp.theta),
            emulator.fit_hyperparameters,
        )

    def test_fit_with_given_hyperparameters_with_fixed_nugget(self):
        """Given an emulator and a set of hyperparameters that includes a value for the
        nugget, when the emulator is fit with the hyperparameters then these are used to
        train the underlying MOGP GaussianProcess object."""

        hyperparameters = MogpHyperparameters(
            corr_length_scales=[0.5, 0.4], process_var=2, nugget=1.0
        )
        for nugget in ["DEFAULT", 2.0, "adaptive", "fit", "pivot"]:
            with self.subTest(nugget=nugget):
                emulator = (
                    MogpEmulator(nugget=nugget) if nugget != "DEFAULT" else MogpEmulator()
                )
                emulator.fit(self.training_data, hyperparameters=hyperparameters)
                self.assertEqual(hyperparameters, emulator.fit_hyperparameters)
                self.assertEqual(
                    MogpHyperparameters.from_mogp_gp_params(emulator.gp.theta),
                    hyperparameters,
                )

    def test_fit_with_given_hyperparameters_without_fixed_nugget(self):
        """Given an emulator and a set of hyperparameters that doesn't include a value for
        the nugget, when the emulator is fit with the hyperparameters then the correlations
        and process variance defined in the hyperparameters are used to train the underlying
        MOGP GaussianProcess object and the nugget used is determined by the settings
        supplied when creating the emulator."""

        hyperparameters = MogpHyperparameters(
            corr_length_scales=[0.5, 0.4], process_var=2
        )
        float_val = 1.0
        for nugget in ["DEFAULT", float_val, "adaptive", "pivot"]:
            with self.subTest(nugget=nugget):
                emulator = (
                    MogpEmulator(nugget=nugget) if nugget != "DEFAULT" else MogpEmulator()
                )
                emulator.fit(self.training_data, hyperparameters=hyperparameters)

                # Check the fitted hyperparameters are as calculated from MOGP
                hyperparameters_gp = MogpHyperparameters.from_mogp_gp_params(
                    emulator.gp.theta
                )
                self.assertEqual(hyperparameters_gp, emulator.fit_hyperparameters)

                # Check the correlation length scale parameters and process variance agree
                # with those supplied for fitting.
                self.assertEqualWithinTolerance(
                    hyperparameters.corr_length_scales,
                    emulator.fit_hyperparameters.corr_length_scales,
                )
                self.assertEqualWithinTolerance(
                    hyperparameters.process_var, emulator.fit_hyperparameters.process_var
                )

                # Check nugget used in fitting agrees with the specific value supplied at
                # emulator creation, if relevant.
                if nugget == float_val:
                    self.assertEqualWithinTolerance(
                        nugget, emulator.fit_hyperparameters.nugget
                    )

    def test_fit_with_given_hyperparameters_missing_nugget_error(self):
        """Given an emulator where the nugget is to be estimated (via the 'fit' value in
        MOGP), raise a ValueError if one attempts to fit the emulator with specific
        hyperparameters that don't include a nugget."""

        emulator = MogpEmulator(nugget="fit")
        hyperparameters = MogpHyperparameters(
            corr_length_scales=[0.5, 0.4], process_var=2
        )
        with self.assertRaisesRegex(
            ValueError,
            exact(
                "The underlying MOGP GaussianProcess was created with 'nugget'='fit', but "
                "the nugget supplied during fitting is "
                f"{hyperparameters.nugget}, when it should instead be a float."
            ),
        ):
            emulator.fit(self.training_data, hyperparameters=hyperparameters)

    def test_predict_returns_prediction_object(self):
        """Given a trained emulator, check that predict() returns a Prediction object."""

        emulator = MogpEmulator()
        emulator.fit(self.training_data)

        self.assertIsInstance(emulator.predict(self.x), Prediction)

    def test_predict_non_input_error(self):
        """Given an emulator, test that a TypeError is raised when trying to predict on
        an object not of type Input."""

        emulator = MogpEmulator()

        for x in [0.5, None, "0.5"]:
            with self.subTest(x=x), self.assertRaisesRegex(
                TypeError,
                exact(f"Expected 'x' to be of type Input, but received {type(x)}."),
            ):
                emulator.predict(x)

    def test_predict_not_trained_error(self):
        """Given an emulator that hasn't been trained on data, test that an
        AssertionError is raised when using the emulator to make a prediction."""

        emulator = MogpEmulator()

        with self.assertRaisesRegex(
            RuntimeError,
            exact(
                "Cannot make prediction because emulator has not been trained on any data."
            ),
        ):
            emulator.predict(self.x)

    def test_predict_input_wrong_dim_error(self):
        """Given a trained emulator, test that a ValueError is raised if the dimension
        of the input is not the same as for the training data."""

        emulator = MogpEmulator()
        emulator.fit(self.training_data)
        expected_dim = len(self.training_data[0].input)

        for x in [Input(), Input(0.5), Input(0.5, 0.5, 0.5)]:
            with self.subTest(x=x), self.assertRaisesRegex(
                ValueError,
                exact(
                    f"Expected 'x' to be an Input with {expected_dim} coordinates, but "
                    f"it has {len(x)} instead."
                ),
            ):
                emulator.predict(x)

    def test_predict_training_data_points(self):
        """Given an emulator trained on data, test that the training data inputs are
        predicted exactly with no uncertainty."""

        emulator = MogpEmulator()
        emulator.fit(self.training_data)
        for datum in self.training_data:
            self.assertEqual(
                Prediction(estimate=datum.output, variance=0),
                emulator.predict(datum.input),
            )

    def test_predict_away_training_data_points(self):
        """Given an emulator trained on data, test that predictions away from the data
        inputs have uncertainty."""

        emulator = MogpEmulator()
        emulator.fit(self.training_data)

        # Note: use list in the following because Inputs aren't hashable
        training_inputs = [datum.input for datum in self.training_data]

        for x in (Input(0.1 * n, 0.1 * n) for n in range(1, 10)):
            assert x not in training_inputs
            self.assertTrue(emulator.predict(x).variance > 0)


class TestMogpHyperparameters(ExauqTestCase):
    def setUp(self) -> None:
        # N.B. although single-element Numpy arrays can be converted to scalars this is
        # deprecated functionality and will throw an error in the future.
        self.nonreal_objects = [2j, "1", np.array([2])]
        self.negative_reals = [-0.5, -math.inf]
        self.nonpositive_reals = self.negative_reals + [0]
        self.hyperparameters = {
            "correlation": {
                "func": MogpHyperparameters.transform_corr,
                "arg": "corr",
            },
            "variance": {
                "func": MogpHyperparameters.transform_cov,
                "arg": "cov",
            },
            "nugget": {
                "func": MogpHyperparameters.transform_nugget,
                "arg": "nugget",
            },
        }

        self.correlations = [[0.1], [0.1, 0.2]]
        self.variances = [1.1, 1.2]
        self.real_nuggets = [0, 2.1, np.float16(2)]
        self.nugget_types = ["fixed", "fit", "adaptive", "pivot"]

    def make_hyperparameters(self, corr=[0.1], cov=1.1, nugget=0):
        """Make hyperparameters, possibly with default values for the correlation length
        scales, process variance and nugget."""
        return MogpHyperparameters(corr, cov, nugget)

    def make_mogp_gp_params(self, corr=[0.1], cov=1.1, nugget=1.0):
        """Make a ``GPParams`` object, possibly with default values for the correlation
        length scales, process variance and nugget. Note: permitted values of arguments
        are either:

        * both `corr` and `cov` are not ``None``; or
        * all three of `corr`, `cov`, `nugget` are ``None`` (in which case an empty
          ``GPParams`` object is returned).
        """

        if all(arg is None for arg in [corr, cov, nugget]):
            return mogp.GPParams.GPParams()

        elif all(arg is not None for arg in [corr, cov, nugget]):
            return self.make_hyperparameters(corr, cov, nugget).to_mogp_gp_params()

        # If here then nugget is None
        elif corr is not None and cov is not None:
            return self.make_hyperparameters(corr, cov, nugget).to_mogp_gp_params(
                "adaptive"
            )

        else:
            raise ValueError(
                "Either all args should be None or 'corr_length_scales' and 'process_var' "
                "should both not be None."
            )

    def test_inherits_from_GaussianProcessHyperparameters(self):
        """MogpHyperparameters inherits from GaussianProcessHyperparameters."""

        self.assertIsInstance(
            self.make_hyperparameters(corr=[0.1], cov=1.1, nugget=0),
            GaussianProcessHyperparameters,
        )

    def test_equals_checks_for_same_type(self):
        """An instance of MogpHyperparameters is not equal to an object of a different
        class."""

        params1 = MogpHyperparameters([1], 1, 1)
        params2 = GaussianProcessHyperparameters([1], 1, 1)
        self.assertNotEqual(params1, params2)
        self.assertNotEqual(params2, params1)

    def test_to_mogp_gp_params_type_error_if_nugget_type_not_str(self):
        """A TypeError is raised if the nugget type is not a string."""

        nugget_types = [1.0, 2j, np.array([2])]
        for nugget_type in nugget_types:
            with self.subTest(nugget_type=nugget_type), self.assertRaisesRegex(
                TypeError,
                exact(
                    f"Expected 'nugget_type' to be of type str, but got {type(nugget_type)}."
                ),
            ):
                _ = self.make_hyperparameters().to_mogp_gp_params(nugget_type=nugget_type)

    def test_to_mogp_gp_params_value_error_if_nugget_type_not_one_of_fit_methods(self):
        """A ValueError is raised if the nugget type is not one of 'fixed', 'fit',
        'adaptive', 'pivot'."""

        with self.assertRaisesRegex(
            ValueError,
            exact(
                "'nugget_type' must be one of {'fixed', 'fit', 'adaptive', 'pivot'}, "
                "but got 'foo'."
            ),
        ):
            _ = self.make_hyperparameters().to_mogp_gp_params(nugget_type="foo")

    def test_to_mogp_gp_params_always_copies_over_corr_and_cov(self):
        """The correlation length scale parameters and process variance are copied over to
        the returned GPParams object."""

        for corr, cov, nugget, nugget_type in itertools.product(
            self.correlations, self.variances, self.real_nuggets, self.nugget_types
        ):
            with self.subTest(corr=corr, cov=cov, nugget=nugget, nugget_type=nugget_type):
                hyperparameters = self.make_hyperparameters(
                    corr=corr, cov=cov, nugget=nugget
                )
                params = hyperparameters.to_mogp_gp_params(nugget_type=nugget_type)
                self.assertEqualWithinTolerance(
                    params.corr, hyperparameters.corr_length_scales
                )
                self.assertEqualWithinTolerance(params.cov, hyperparameters.process_var)

    def test_to_mogp_gp_params_sets_nugget_when_type_fixed_or_fit(self):
        """When the `nugget_type` is one of 'fixed' or 'fit' and the nugget is defined as
        a real number, then the nugget is copied over to the returned GPParams object."""

        for nugget, nugget_type in itertools.product(self.real_nuggets, ["fixed", "fit"]):
            with self.subTest(nugget=nugget, nugget_type=nugget_type):
                hyperparameters = self.make_hyperparameters(nugget=nugget)
                params = hyperparameters.to_mogp_gp_params(nugget_type=nugget_type)
                self.assertEqualWithinTolerance(params.nugget, hyperparameters.nugget)
                self.assertEqual(nugget_type, params.nugget_type)

    def test_to_mogp_gp_params_value_error_when_type_fixed_or_fit_and_nugget_none(self):
        """When the `nugget_type` is one of 'fixed' or 'fit', then a ValueError is raised
        if the nugget is None."""

        hyperparameters = self.make_hyperparameters(nugget=None)
        for nugget_type in ["fixed", "fit"]:
            with self.subTest(nugget_type=nugget_type), self.assertRaisesRegex(
                ValueError,
                exact(
                    f"Cannot set nugget fitting method to 'nugget_type = {nugget_type}' "
                    "when this object's nugget is None."
                ),
            ):
                _ = hyperparameters.to_mogp_gp_params(nugget_type=nugget_type)

    def test_to_mogp_gp_params_sets_nugget_type_when_adaptive_or_pivot(self):
        """When the `nugget_type` is 'adaptive' or 'pivot', then the nugget fitting
        method in the output GPParams object is set to this and the value of the nugget is
        not copied over to the output."""

        for nugget, nugget_type in itertools.product([1.0, None], ["adaptive", "pivot"]):
            with self.subTest(nugget=nugget, nugget_type=nugget_type):
                hyperparameters = self.make_hyperparameters(nugget=nugget)
                params = hyperparameters.to_mogp_gp_params(nugget_type=nugget_type)
                self.assertEqual(nugget_type, params.nugget_type)
                self.assertIsNone(params.nugget)

    def test_from_mogp_gp_params_inverse_of_to_mogp_gp_params(self):
        """Given some hyperparameters, creating a GPParams from them and then creating
        hyperparameters from the result gives the same hyperparameters as we started
        with."""

        nuggets = self.real_nuggets + [None]
        for corr, cov, nugget in itertools.product(
            self.correlations, self.variances, nuggets
        ):
            with self.subTest(corr=corr, cov=cov, nugget=nugget):
                params = self.make_mogp_gp_params(corr, cov, nugget)
                hyperparameters = MogpHyperparameters.from_mogp_gp_params(params)
                expected = self.make_hyperparameters(corr, cov, nugget)
                self.assertEqual(expected, hyperparameters)

    def test_from_mogp_gp_params_arg_type_check(self):
        """A TypeError is raised if the argument passed is not a GPParams object."""

        params = "foo"
        with self.assertRaisesRegex(
            TypeError,
            exact(
                "Expected 'params' to be of type mogp_emulator.GPParams.GPParams, but "
                f"received {type(params)}."
            ),
        ):
            _ = MogpHyperparameters.from_mogp_gp_params("foo")

    def test_from_mogp_gp_params_arg_corr_and_cov_must_not_be_none(self):
        """A ValueError is raised if the argument is a GPParams object with the
        correlations and the process variance being None."""

        with self.assertRaisesRegex(
            ValueError,
            exact(
                "Cannot create hyperparameters with correlation length scales and process "
                "variance equal to None in 'params'."
            ),
        ):
            params = self.make_mogp_gp_params(corr=None, cov=None, nugget=None)
            _ = MogpHyperparameters.from_mogp_gp_params(params)


if __name__ == "__main__":
    unittest.main()
