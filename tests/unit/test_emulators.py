import itertools
import math
import unittest
import unittest.mock

import mogp_emulator as mogp
import numpy as np

from exauq.core.emulators import MogpEmulator, MogpHyperparameters
from exauq.core.modelling import Input, Prediction, TrainingDatum
from exauq.core.numerics import equal_within_tolerance
from tests.utilities.utilities import exact


class TestMogpEmulator(unittest.TestCase):
    def setUp(self) -> None:
        # Some default args to use for constructing mogp GaussianProcess objects
        self.inputs = np.array([[0, 0], [0.2, 0.1]])
        self.targets = np.array([1, 2])

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

        emulator = MogpEmulator(inputs=self.inputs, targets=self.targets)
        self.assertEqual(0, emulator.gp.inputs.size)
        self.assertEqual(0, emulator.gp.targets.size)

    def test_training_data(self):
        """Test that the training data in the underlying GP and the
        training_data property is empty when an emulator is constructed."""

        emulator = MogpEmulator()
        self.assertEqual(0, emulator.gp.inputs.size)
        self.assertEqual(0, emulator.gp.targets.size)
        self.assertEqual([], emulator.training_data)

    def test_fit_no_training_data(self):
        """Test that fitting the emulator results in the underlying GP being fit
        with hyperparameter estimation."""

        inputs = np.array([[0, 0], [0.2, 0.1], [0.3, 0.5], [0.7, 0.4], [0.9, 0.8]])
        targets = np.array([1, 2, 3.1, 9, 2])
        gp = mogp.fit_GP_MAP(mogp.GaussianProcess(inputs, targets))
        emulator = MogpEmulator()
        emulator.fit(TrainingDatum.list_from_arrays(inputs, targets))

        # Note: need to use allclose because fitting is not deterministic.
        self.assertTrue(
            np.allclose(gp.theta.corr, emulator.gp.theta.corr, rtol=1e-5, atol=0)
        )
        self.assertTrue(
            np.allclose(gp.theta.cov, emulator.gp.theta.cov, rtol=1e-5, atol=0)
        )
        self.assertTrue(
            np.allclose(gp.theta.nugget, emulator.gp.theta.nugget, rtol=1e-5, atol=0)
        )

    def test_fit_bounds_converts_to_raw(self):
        """Test that bounds on correlation length parameters and the covariance
        are transformed to bounds on raw parameters when fitting. For the
        transformations, see:
        https://mogp-emulator.readthedocs.io/en/latest/implementation/GPParams.html#mogp_emulator.GPParams.GPParams
        """

        mock = unittest.mock.MagicMock()
        with unittest.mock.patch("exauq.core.emulators.fit_GP_MAP", mock):
            bounds = (
                (1, math.e),  # correlation = exp(-0.5 * raw_corr)
                (math.exp(-2), math.exp(3)),  # correlation = exp(-0.5 * raw_corr)
                (1, math.e),  # covariance
            )
            emulator = MogpEmulator()
            emulator.fit(self.training_data, hyperparameter_bounds=bounds)

            # Check that the underlying fitting function was called with correct
            # bounds
            raw_bounds = (
                (-2.0, 0.0),  # raw correlation = -2 * log(corr)
                (-6.0, 4.0),  # raw correlation
                (0.0, 1.0),  # raw covariance = log(cov)
            )
            self.assertEqual(raw_bounds, mock.call_args.kwargs["bounds"])

    def test_fit_bounds_none(self):
        """Test that None entries for bounds are converted to None entries when
        fitting."""

        mock = unittest.mock.MagicMock()
        with unittest.mock.patch("exauq.core.emulators.fit_GP_MAP", mock):
            bounds = ((1, None), (None, 1))
            emulator = MogpEmulator()
            emulator.fit(self.training_data, hyperparameter_bounds=bounds)

            # Check that the underlying fitting function was called with correct
            # bounds
            raw_bounds = ((None, 0.0), (None, 0.0))
            self.assertEqual(raw_bounds, mock.call_args.kwargs["bounds"])

    def test_compute_raw_param_bounds_non_positive(self):
        """Test that non-positive lower bounds are converted to None when
        fitting."""

        mock = unittest.mock.MagicMock()
        with unittest.mock.patch("exauq.core.emulators.fit_GP_MAP", mock):
            bounds = ((-np.inf, 1), (0, 1), (-math.e, 1))
            emulator = MogpEmulator()
            emulator.fit(self.training_data, hyperparameter_bounds=bounds)

            # Check that the underlying fitting function was called with correct
            # bounds
            raw_bounds = ((0.0, None), (0.0, None), (None, 0.0))
            self.assertEqual(raw_bounds, mock.call_args.kwargs["bounds"])

    def test_fit_training_data(self):
        """Test that the emulator is trained on the supplied training data
        and its training_data property is updated."""

        inputs = np.array([[0, 0], [0.2, 0.1], [0.3, 0.5], [0.7, 0.4], [0.9, 0.8]])
        targets = np.array([1, 2, 3.1, 9, 2])
        emulator = MogpEmulator()
        training_data = TrainingDatum.list_from_arrays(inputs, targets)
        emulator.fit(training_data)

        self.assertTrue(np.allclose(inputs, emulator.gp.inputs))
        self.assertTrue(np.allclose(targets, emulator.gp.targets))
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

        inputs = np.array([[0, 0], [0.2, 0.1], [0.3, 0.5], [0.7, 0.4], [0.9, 0.8]])
        targets = np.array([1, 2, 3.1, 9, 2])
        gp = mogp.fit_GP_MAP(mogp.GaussianProcess(inputs, targets))

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
            TrainingDatum.list_from_arrays(inputs, targets),
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

    def test_fit_gp_kwargs(self):
        """Test that, after fitting, the underlying mogp emulator has the same
        kwarg settings as the original did before fitting."""

        emulator = MogpEmulator(nugget=self.gp_kwargs["nugget"])

        emulator.fit(training_data=self.training_data)

        self.assertEqual(self.gp_kwargs["nugget"], emulator.gp.nugget_type)

    def test_fit_ignores_inputs_targets(self):
        """Test that fitting ignores any inputs and targets supplied during
        initialisation of the emulator."""

        emulator = MogpEmulator(inputs=self.inputs, targets=self.targets)
        emulator.fit([])

        self.assertEqual(0, emulator.gp.inputs.size)
        self.assertEqual(0, emulator.gp.targets.size)
        self.assertEqual([], emulator.training_data)

    def test_fit_empty_training_data(self):
        """Test that the training data and underlying mogp GaussianProcess
        are not changed when fitting with no training data."""

        # Case where no data has previously been fit
        emulator = MogpEmulator()

        emulator.fit([])

        self.assertEqual(0, emulator.gp.inputs.size)
        self.assertEqual(0, emulator.gp.targets.size)
        self.assertEqual([], emulator.training_data)

        # Case where data has previously been fit
        inputs = np.array([[0, 0], [0.2, 0.1], [0.3, 0.5], [0.7, 0.4], [0.9, 0.8]])
        targets = np.array([1, 2, 3.1, 9, 2])
        emulator.fit(TrainingDatum.list_from_arrays(inputs, targets))
        expected_inputs = emulator.gp.inputs
        expected_targets = emulator.gp.targets
        expected_training_data = emulator.training_data

        emulator.fit([])

        self.assertTrue(np.allclose(expected_inputs, emulator.gp.inputs))
        self.assertTrue(np.allclose(expected_targets, emulator.gp.targets))
        self.assertEqual(expected_training_data, emulator.training_data)

    def test_fitted_hyperparameters_can_be_retrieved(self):
        """Given an emulator, when it is fit to training data then the fitted
        hyperparameters can be retrieved and agree with those in the underlying
        MOGP GaussianProcess object."""

        emulator = MogpEmulator()
        emulator.fit(self.training_data)
        self.assertEqual(
            MogpHyperparameters.from_mogp_gp(emulator.gp), emulator.fit_hyperparameters
        )

    def test_fit_with_given_hyperparameters_with_fixed_nugget(self):
        """Given an emulator and a set of hyperparameters that includes a value for the
        nugget, when the emulator is fit with the hyperparameters then these are used to
        train the underlying MOGP GaussianProcess object."""

        hyperparameters = MogpHyperparameters(corr=[0.5, 0.4], cov=2, nugget=1.0)
        for nugget in [2.0, "adaptive", "fit", "pivot"]:
            with self.subTest(nugget=nugget):
                emulator = MogpEmulator(nugget=nugget)
                emulator.fit(self.training_data, hyperparameters=hyperparameters)
                self.assertEqual(hyperparameters, emulator.fit_hyperparameters)
                self.assertEqual(
                    MogpHyperparameters.from_mogp_gp(emulator.gp), hyperparameters
                )

    def test_fit_with_given_hyperparameters_without_fixed_nugget(self):
        """Given an emulator and a set of hyperparameters that doesn't include a value for
        the nugget, when the emulator is fit with the hyperparameters then these are used
        to train the underlying MOGP GaussianProcess object and the nugget is determined
        as per the settings supplied when creating the emulator."""

        hyperparameters = MogpHyperparameters(corr=[0.5, 0.4], cov=2)
        float_val = 1.0
        for nugget in [float_val, "adaptive", "pivot"]:
            with self.subTest(nugget=nugget):
                emulator = MogpEmulator(nugget=nugget)
                emulator.fit(self.training_data, hyperparameters=hyperparameters)

                # Check the fitted hyperparameters are as calculated from MOGP
                hyperparameters_gp = MogpHyperparameters.from_mogp_gp(emulator.gp)

                # Check the correlation length parameters and covariance agree with those
                # supplied for fitting.
                self.assertEqual(hyperparameters_gp, emulator.fit_hyperparameters)
                self.assertTrue(
                    equal_within_tolerance(
                        hyperparameters.corr, emulator.fit_hyperparameters.corr
                    )
                )
                self.assertTrue(
                    equal_within_tolerance(
                        hyperparameters.cov, emulator.fit_hyperparameters.cov
                    )
                )

                # Check nugget used in fitting agrees with the specific value supplied at
                # emulator creation, if relevant.
                if nugget == float_val:
                    self.assertTrue(
                        equal_within_tolerance(
                            nugget, emulator.fit_hyperparameters.nugget
                        )
                    )

    def test_fit_with_given_hyperparameters_missing_nugget_error(self):
        """Given an emulator where the nugget is to be estimated (via the 'fit' value in
        MOGP), raise a ValueError if one attempts to fit the emulator with specific
        hyperparameters that don't include a nugget."""

        emulator = MogpEmulator(nugget="fit")
        hyperparameters = MogpHyperparameters(corr=[0.5, 0.4], cov=2)
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


class TestMogpHyperparameters(unittest.TestCase):
    def setUp(self) -> None:
        self.positive_reals = [0.1, 1, 10, np.float16(1.1)]

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
            "covariance": {
                "func": MogpHyperparameters.transform_cov,
                "arg": "cov",
            },
            "nugget": {
                "func": MogpHyperparameters.transform_nugget,
                "arg": "nugget",
            },
        }

        self.correlations = [[0.1], [0.1, 0.2]]
        self.covariances = [1.1, 1.2]
        self.real_nuggets = [0, 2.1, np.float16(2)]
        self.nugget_types_strs = ["adaptive", "fit", "pivot"]
        self.nugget_types = [0, 1.0, np.float16(1.0)] + self.nugget_types_strs

    def assertEqualWithinTolerance(self, x1, x2):
        """Test for equality using the `numerics.equal_within_tolerance` function."""

        return self.assertTrue(
            equal_within_tolerance(x1, x2),
            msg=f"Values {x1} and {x2} not equal within tolerance.",
        )

    def make_hyperparameters(self, corr=[0.1], cov=1.1, nugget=0):
        """Make hyperparameters, possibly with default values for the correlations,
        covariance and nugget."""
        return MogpHyperparameters(corr, cov, nugget)

    def make_mogp_gp_params(self, corr=[0.1], cov=1.1, nugget=1.0):
        """Make a ``GPParams`` object, possibly with default values for the correlations,
        covariance and nugget. Note: permitted values of arguments are either:

        * both `corr` and `cov` are not ``None``; or
        * all three of `corr`, `cov`, `nugget` are ``None`` (in which case an empty
          ``GPParams`` object is returned).
        """

        if all(arg is None for arg in [corr, cov, nugget]):
            return mogp.GPParams.GPParams()

        elif corr is not None and cov is not None:
            nugget_type = "adaptive" if nugget is None else "fit"
            return self.make_hyperparameters(corr, cov, nugget).to_mogp_gp_params(
                nugget_type
            )

        else:
            raise ValueError(
                "Either all args should be None or 'corr' and 'cov' should both not be None."
            )

    def test_init_checks_arg_types(self):
        """A TypeError is raised upon initialisation if:

        * the correlations is not a sequence; or
        * the covariance is not a real number; or
        * the nugget is not ``None`` or a real number.
        """

        # correlations
        nonseq_objects = [1.0, {1.0}]
        for corr in nonseq_objects:
            with self.subTest(corr=corr), self.assertRaisesRegex(
                TypeError,
                exact(
                    f"Expected 'corr' to be a sequence or array, but received {type(corr)}."
                ),
            ):
                _ = MogpHyperparameters(corr=corr, cov=1.0, nugget=1.0)

        # covariance
        nonreal_objects = self.nonreal_objects + [None]
        for cov in nonreal_objects:
            with self.subTest(cov=cov), self.assertRaisesRegex(
                TypeError,
                exact(f"Expected 'cov' to be a real number, but received {type(cov)}."),
            ):
                _ = MogpHyperparameters(corr=[1.0], cov=cov, nugget=1.0)

        # nugget
        for nugget in self.nonreal_objects:
            with self.subTest(nugget=nugget), self.assertRaisesRegex(
                TypeError,
                exact(
                    f"Expected 'nugget' to be a real number, but received {type(nugget)}."
                ),
            ):
                _ = MogpHyperparameters(corr=[1.0], cov=1.0, nugget=nugget)

    def test_init_checks_arg_values(self):
        """A ValueError is raised upon initialisation if:

        * the correlation is not a sequence / array of positive real numbers; or
        * the covariance is not a positive real number; or
        * the nugget is < 0 (if not None).
        """

        # correlations
        bad_values = [[x] for x in self.nonreal_objects + self.nonpositive_reals]
        for corr in bad_values:
            with self.subTest(corr=corr), self.assertRaisesRegex(
                ValueError,
                exact(
                    "Expected 'corr' to be a sequence or array of positive real numbers, "
                    f"but found element {corr[0]} of type {type(corr[0])}."
                ),
            ):
                _ = MogpHyperparameters(corr=corr, cov=1.0, nugget=1.0)

        # covariance
        for cov in self.nonpositive_reals:
            with self.subTest(cov=cov), self.assertRaisesRegex(
                ValueError,
                exact(
                    f"Expected 'cov' to be a positive real number, but received {cov}."
                ),
            ):
                _ = MogpHyperparameters(corr=[1.0], cov=cov, nugget=1.0)

        # nugget
        for nugget in self.negative_reals:
            with self.subTest(nugget=nugget), self.assertRaisesRegex(
                ValueError,
                exact(
                    f"Expected 'nugget' to be a positive real number, but received {nugget}."
                ),
            ):
                _ = MogpHyperparameters(corr=[1.0], cov=1.0, nugget=nugget)

    def test_transformation_formulae(self):
        """The transformed correlation is equal to `-2 * log(corr)`.
        The transformed covariance is equal to `log(cov)`.
        The transformed nugget is equal to `log(nugget)`."""

        for x in self.positive_reals:
            with self.subTest(hyperparameter="correlation", x=x):
                transformation_func = self.hyperparameters["correlation"]["func"]
                self.assertEqual(-2 * math.log(x), transformation_func(x))

            with self.subTest(hyperparameter="covariance", x=x):
                transformation_func = self.hyperparameters["covariance"]["func"]
                self.assertEqual(math.log(x), transformation_func(x))

            with self.subTest(hyperparameter="nugget", x=x):
                transformation_func = self.hyperparameters["nugget"]["func"]
                self.assertEqual(math.log(x), transformation_func(x))

    def test_transformation_of_infinity(self):
        """The transformed correlation of `inf` is equal to `-inf`.
        The transformed covariance of `inf` is equal to `inf`."""

        with self.subTest(hyperparameter="correlation"):
            transformation_func = self.hyperparameters["correlation"]["func"]
            self.assertEqual(-math.inf, transformation_func(math.inf))

        with self.subTest(hyperparameter="covariance"):
            transformation_func = self.hyperparameters["covariance"]["func"]
            self.assertEqual(math.inf, transformation_func(math.inf))

        with self.subTest(hyperparameter="nugget"):
            transformation_func = self.hyperparameters["nugget"]["func"]
            self.assertEqual(math.inf, transformation_func(math.inf))

    def test_transforms_non_real_arg_raises_type_error(self):
        "A TypeError is raised if the argument supplied is not a real number."

        for hyperparameter, x in itertools.product(
            self.hyperparameters, self.nonreal_objects
        ):
            arg = self.hyperparameters[hyperparameter]["arg"]
            transformation_func = self.hyperparameters[hyperparameter]["func"]
            with self.subTest(hyperparameter=hyperparameter, x=x), self.assertRaisesRegex(
                TypeError,
                exact(f"Expected '{arg}' to be a real number, but received {type(x)}."),
            ):
                _ = transformation_func(x)

    def test_transforms_with_nonpositive_value_raises_value_error(self):
        "A ValueError is raised if the argument supplied is not > 0."

        for hyperparameter, x in itertools.product(
            self.hyperparameters, self.nonpositive_reals
        ):
            arg = self.hyperparameters[hyperparameter]["arg"]
            transformation_func = self.hyperparameters[hyperparameter]["func"]
            with self.subTest(hyperparameter=hyperparameter, x=x), self.assertRaisesRegex(
                ValueError,
                exact(f"'{arg}' must be a positive real number, but received {x}."),
            ):
                _ = transformation_func(x)

    def test_to_mogp_gp_params_type_error_if_nugget_type_not_real_number_or_str(self):
        """A TypeError is raised if the nugget type is not a real number or a string."""

        nugget_type = [2j, np.array([2])]
        with self.assertRaisesRegex(
            TypeError,
            exact(
                "Expected 'nugget_type' to be of type str of float, but got "
                f"{type(nugget_type)}."
            ),
        ):
            _ = self.make_hyperparameters().to_mogp_gp_params(nugget_type=nugget_type)

    def test_to_mogp_gp_params_value_error_if_nugget_type_not_positive_real_or_one_of_fit_methods(
        self,
    ):
        """A ValueError is raised if the nugget type is a negative real number or one of
        'fit', 'adaptive', 'pivot'."""

        nugget_types = [-1, "foo"]
        for nugget_type in nugget_types:
            with self.subTest(nugget_type=nugget_type), self.assertRaisesRegex(
                ValueError,
                exact(
                    "'nugget_type' must be a real number >= 0 or one of "
                    "{'adaptive', 'fit', 'pivot'}, but got " + f"{nugget_type}."
                ),
            ):
                _ = self.make_hyperparameters().to_mogp_gp_params(nugget_type=nugget_type)

    def test_to_mogp_gp_params_always_copies_over_corr_and_cov(self):
        """The correlation length parameters and covariance are copied over to the
        returned GPParams object."""

        for corr, cov, nugget, nugget_type in itertools.product(
            self.correlations, self.covariances, self.real_nuggets, self.nugget_types
        ):
            with self.subTest(corr=corr, cov=cov, nugget=nugget, nugget_type=nugget_type):
                hyperparameters = self.make_hyperparameters(
                    corr=corr, cov=cov, nugget=nugget
                )
                params = hyperparameters.to_mogp_gp_params(nugget_type=nugget_type)
                self.assertEqualWithinTolerance(params.corr, hyperparameters.corr)
                self.assertEqualWithinTolerance(params.cov, hyperparameters.cov)

    def test_to_mogp_gp_params_sets_nugget_when_a_float(self):
        """When the nugget is set as a floating point value, then it is copied over to
        the returned GPParams object, regardless of the value of the `nugget_type`
        argument."""

        for nugget, nugget_type in itertools.product(
            self.real_nuggets, self.nugget_types
        ):
            with self.subTest(nugget=nugget, nugget_type=nugget_type):
                hyperparameters = self.make_hyperparameters(nugget=nugget)
                params = hyperparameters.to_mogp_gp_params(nugget_type=nugget_type)
                self.assertEqualWithinTolerance(params.nugget, hyperparameters.nugget)

    def test_to_mogp_gp_params_nugget_set_as_specified_if_stored_nugget_is_none(self):
        """The value of `nugget_type` is used to set the nugget in the returned GPParams
        object if no nugget is stored in the hyperparameters and `nugget_type` is not equal
        to 'fit'."""

        hyperparameters = self.make_hyperparameters(nugget=None)
        for nugget_type in ["adaptive", "pivot"]:
            with self.subTest(nugget_type=nugget_type):
                params = hyperparameters.to_mogp_gp_params(nugget_type=nugget_type)
                self.assertEqual(nugget_type, params.nugget_type)
                self.assertIsNone(params.nugget)

        nugget_type = 1.0
        with self.subTest(nugget_type=nugget_type):
            params = hyperparameters.to_mogp_gp_params(nugget_type=nugget_type)
            self.assertEqual("fixed", params.nugget_type)
            self.assertEqual(nugget_type, params.nugget)

    def test_to_mogp_gp_params_cannot_set_to_fit_if_stored_nugget_none(self):
        """Using a value of 'fit' for `nugget_type` when no nugget is stored in the
        hyperparameters raises a ValueError."""

        hyperparameters = self.make_hyperparameters(nugget=None)
        with self.assertRaisesRegex(
            ValueError,
            exact(
                "Cannot specify 'nugget_type' to be 'fit' with this object's "
                "nugget hyperparameter set to None."
            ),
        ):
            _ = hyperparameters.to_mogp_gp_params(nugget_type="fit")

    def test_from_mogp_gp_params_inverse_of_to_mogp_gp_params(self):
        """Given some hyperparameters, creating a GPParams from them and then creating
        hyperparameters from the result gives the same hyperparameters as we started
        with."""

        nuggets = self.real_nuggets + [None]
        for corr, cov, nugget in itertools.product(
            self.correlations, self.covariances, nuggets
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
        correlations and the covariance being None."""

        with self.assertRaisesRegex(
            ValueError,
            exact(
                "Cannot create hyperparameters with correlations and covariance equal to "
                "None in 'params'."
            ),
        ):
            params = self.make_mogp_gp_params(corr=None, cov=None, nugget=None)
            _ = MogpHyperparameters.from_mogp_gp_params(params)


if __name__ == "__main__":
    unittest.main()
