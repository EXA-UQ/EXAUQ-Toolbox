import math
import unittest
import unittest.mock

import mogp_emulator as mogp
import numpy as np

from exauq.core.emulators import MogpEmulator, MogpHyperparameters
from exauq.core.modelling import Input, Prediction, TrainingDatum
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

    def test_fit_with_hyperparams(self):
        """Given an emulator and a set of hyperparameters, when the emulator is
        fit with the hyperparameters then these can be retrieved as the
        hyperparameters used in the last fitting."""

        emulator = MogpEmulator()
        hyperparameters = MogpHyperparameters(
            mean=[5.0, 6.0], corr=[0.5, 0.4, 0.6], cov=2, nugget_type="fixed", nugget=1.0
        )

        training_data = TrainingDatum.list_from_arrays(
            np.array([[0, 0], [0.2, 0.1], [0.3, 0.5], [0.7, 0.4], [0.9, 0.8]]),
            np.array([1, 2, 3.1, 9, 2]),
        )
        emulator.fit(training_data, hyperparameters=hyperparameters)

        self.assertEqual(hyperparameters, emulator.fit_hyperparameters)

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


if __name__ == "__main__":
    unittest.main()
