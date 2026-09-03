import unittest

import numpy as np

from exauq.core.modelling import (
    GaussianProcessHyperparameters,
    GaussianProcessPrediction,
    Input,
    MultiLevel,
    TrainingDatum,
)
from tests.utilities.utilities import ExauqTestCase, exact

try:
    from exauq.core.bayhem import BayHEMGP, BayHEMGPHyperparameters
except ImportError:  # the 'bayhem' extra (pymc) is not installed, e.g. Python < 3.12
    BayHEMGP = BayHEMGPHyperparameters = None


@unittest.skipIf(BayHEMGP is None, "requires the 'bayhem' extra (pymc, Python 3.12+)")
class TestBayHEM(ExauqTestCase):
    def setUp(self) -> None:
        # Some default args to use for constructing BayHEM objects
        self.training_data = MultiLevel(
            {
                1: (
                    TrainingDatum(Input(0.1, 0.2), 1),
                    TrainingDatum(Input(0.3, 0.4), 2),
                    TrainingDatum(Input(0.5, 0.6), 3),
                    TrainingDatum(Input(0.7, 0.8), 4),
                ),
                2: (
                    TrainingDatum(Input(0.11, 0.21), 1.1),
                    TrainingDatum(Input(0.31, 0.41), 2.1),
                    TrainingDatum(Input(0.51, 0.61), 3.1),
                ),
                3: (TrainingDatum(Input(0, 1), 0),),
            }
        )

        # Input not contained in training data for making predictions
        self.x = Input(0.1, 0.1)

        # Setting up emulator
        self.gp = BayHEMGP()

        # Dummy MAP estimate
        self.gp._MAP = {
            "ls1_L1": np.array(0.5),
            "ls2_L1": np.array(0.2),
            "sig_L1": np.array(1.5),
            "nug_L1": np.array(0.01),
            "ls1_L2": np.array(0.75),
            "sig_L3": np.array(2.5),
            "nug_L3": np.array(0.001),
        }

    def test_error_hyperparameters_no_inputs_levels(self):
        """Tests whether get correct error if attempt to create hyperparameters when
        input dimensions or levels are unknown."""

        msg = "Cannot create hyperparameters before fitting. Input dimensions and levels unknown."

        with self.assertRaisesRegex(ValueError, exact(msg)):
            self.gp._create_default_hyperparameters()

    def test_error_hyperparameters_no_inputs_dims(self):
        """Tests whether get correct error if attempt to create hyperparameters when
        input dimensions are unknown."""

        self.gp._levels = 3

        msg = "Cannot create hyperparameters before fitting. Input dimensions and levels unknown."

        with self.assertRaisesRegex(ValueError, exact(msg)):
            self.gp._create_default_hyperparameters()

    def test_error_hyperparameters_no_levels(self):
        """Tests whether get correct error if attempt to create hyperparameters when
        levels are unknown."""

        self.gp._input_dims = 2

        msg = "Cannot create hyperparameters before fitting. Input dimensions and levels unknown."

        with self.assertRaisesRegex(ValueError, exact(msg)):
            self.gp._create_default_hyperparameters()

    def test_correct_default_hyperparameter_inputs_levels(self):
        """Checking that the number of levels and input dimensions is consistent between the
        training data and default hyperparameters."""

        self.gp._levels = 3
        self.gp._input_dims = 2
        hparams = self.gp._create_default_hyperparameters()

        self.assertEqual(self.gp._levels, hparams.levels)
        self.assertEqual(self.gp._input_dims, hparams.input_dims)

    def test_error_mismatch_hyperparameter_training_levels(self):
        """Checking that the correct error is generated when passing a set of hyperparameters with
        a different number of levels from the training data."""

        hparams = BayHEMGPHyperparameters(input_dims=2, levels=2)
        msg = "Expected 3 levels in hyperparameters, but received 2."

        with self.assertRaisesRegex(ValueError, exact(msg)):
            self.gp.fit(self.training_data, hparams)

        hparams = BayHEMGPHyperparameters(input_dims=2, levels=1)
        msg = "Expected 3 levels in hyperparameters, but received 1."

        with self.assertRaisesRegex(ValueError, exact(msg)):
            self.gp.fit(self.training_data, hparams)

    def test_error_mismatch_hyperparameter_training_input_dims(self):
        """Checking that the correct error is generated when passing a set of hyperparameters with
        a different number of input dimensions from the training data."""

        hparams = BayHEMGPHyperparameters(input_dims=3, levels=3)
        msg = "Expected 2 input dimensions in hyperparameters, but received 3."

        with self.assertRaisesRegex(ValueError, exact(msg)):
            self.gp.fit(self.training_data, hparams)

        hparams = BayHEMGPHyperparameters(input_dims=1, levels=3)
        msg = "Expected 2 input dimensions in hyperparameters, but received 1."

        with self.assertRaisesRegex(ValueError, exact(msg)):
            self.gp.fit(self.training_data, hparams)

    def test_BayHEM_input_data_format(self):
        """Checking that the correct error is generated if have the wrong type
        of training data."""

        training_new = np.array(0)

        with self.assertRaisesRegex(
            TypeError,
            exact(
                f"Expected 'training_data' to be of type {MultiLevel.__name__}, "
                f"but received {type(training_new)} instead."
            ),
        ):
            self.gp.fit(training_new)

    def test_BayHEM_input_all_levels(self):
        """Checking that the training data includes all levels when fitting a
        BayHEM model."""

        training_new = MultiLevel(
            {
                1: (
                    TrainingDatum(Input(0.1, 0.2), 1),
                    TrainingDatum(Input(0.3, 0.4), 2),
                    TrainingDatum(Input(0.5, 0.6), 3),
                    TrainingDatum(Input(0.7, 0.8), 4),
                ),
                4: (TrainingDatum(Input(0, 1), 0),),
            }
        )

        msg = (
            "Missing training data for levels: {2, 3}. Required continuous sequence "
            "from 1 to 4"
        )

        with self.assertRaisesRegex(ValueError, exact(msg)):
            self.gp.fit(training_new)

    def test_BayHEM_empty_level_one(self):
        """Checking that the correct error is given if a training dataset with an empty
        level one is passed to BayHEM.fit."""

        training_new = MultiLevel(
            {
                1: (
                    TrainingDatum(Input(0.1, 0.2), 1),
                    TrainingDatum(Input(0.3, 0.4), 2),
                    TrainingDatum(Input(0.5, 0.6), 3),
                    TrainingDatum(Input(0.7, 0.8), 4),
                ),
                2: (),
                3: (TrainingDatum(Input(0, 1), 0),),
            }
        )

        msg = "Training data must contain at least one point at level 2"

        with self.assertRaisesRegex(ValueError, exact(msg)):
            self.gp.fit(training_new)

    def test_BayHEM_empty_level_higher(self):
        """Checking that the correct error is given if a training dataset with an empty
        level (greater than 1) is passed to BayHEM.fit."""

        training_new = MultiLevel(
            {
                1: (),
                2: (
                    TrainingDatum(Input(0.11, 0.21), 1.1),
                    TrainingDatum(Input(0.31, 0.41), 2.1),
                    TrainingDatum(Input(0.51, 0.61), 3.1),
                ),
                3: (TrainingDatum(Input(0, 1), 0),),
            }
        )

        msg = "Training data must contain at least one point at level 1"

        with self.assertRaisesRegex(ValueError, exact(msg)):
            self.gp.fit(training_new)

    def test_BayHEM_inconsistent_input_dims(self):
        """Checking that the correct error is given if a training dataset with inconsistent
        input dimensions is passed to BayHEM.fit."""

        training_new = MultiLevel(
            {
                1: (
                    TrainingDatum(Input(0.1, 0.2), 1),
                    TrainingDatum(Input(0.3, 0.4), 2),
                    TrainingDatum(Input(0.5, 0.6), 3),
                    TrainingDatum(Input(0.7, 0.8), 4),
                ),
                2: (
                    TrainingDatum(Input(0.11, 0.21, 0.31), 1.1),
                    TrainingDatum(Input(0.31, 0.41, 0.51), 2.1),
                    TrainingDatum(Input(0.51, 0.61, 0.71), 3.1),
                ),
                3: (TrainingDatum(Input(0, 1), 0),),
            }
        )

        msg = "Inconsistent input dimensions. Expected 2 but found 3 at level 2"

        with self.assertRaisesRegex(ValueError, exact(msg)):
            self.gp.fit(training_new)

    def test_extract_hyperparameters_from_MAP(self):
        """Checking that multilevel hyperparameters are correctly inherited across
        levels when extracting from PyMC output."""

        self.gp._input_dims = 2
        self.gp._levels = 3

        expected1 = GaussianProcessHyperparameters(
            corr_length_scales=np.array([0.5, 0.2]), process_var=1.5**2, nugget=0.01
        )

        expected2 = GaussianProcessHyperparameters(
            corr_length_scales=np.array([0.75, 0.2]), process_var=1.5**2, nugget=0.01
        )

        expected3 = GaussianProcessHyperparameters(
            corr_length_scales=np.array([0.75, 0.2]), process_var=2.5**2, nugget=0.001
        )

        self.assertEqual(expected1, self.gp._extract_hyperparameters_from_MAP(level=1))
        self.assertEqual(expected2, self.gp._extract_hyperparameters_from_MAP(level=2))
        self.assertEqual(expected3, self.gp._extract_hyperparameters_from_MAP(level=3))

    def test_BayHEM_predict_not_fitted(self):
        """Checking that the correct error is generated if attempt to predict without
        first fitting the model."""

        gp = BayHEMGP()
        msg = "Model has not been fitted yet. Call fit() first."

        with self.assertRaisesRegex(ValueError, exact(msg)):
            gp.predict(self.x)

    def test_BayHEM_predict_incorrect_data(self):
        """Checking that the correct error is generated if attempt to predict at a new
        point with incorrect type."""

        self.gp._input_dims = 2
        x_new = np.array(0)

        with self.assertRaisesRegex(
            TypeError,
            exact(f"Expected 'x' to be of type Input, but received {type(x_new)}"),
        ):
            self.gp.predict(x_new)

    def test_BayHEM_predict_incorrect_dimensions(self):
        """Checking that the correct error is generated if attempt to predict at a new
        point with incorrect dimensions."""

        self.gp._input_dims = 2
        x_new = Input(0)

        msg = "Expected input of dimension 2, but received 1"

        with self.assertRaisesRegex(ValueError, exact(msg)):
            self.gp.predict(x_new)

    def test_correlation_uses_top_level_length_scales(self):
        """Correlation is the squared-exponential kernel with the top level's fitted
        length scales, and is empty for empty inputs or an unfitted emulator."""

        self.gp._levels = 2
        self.gp._input_dims = 2
        self.gp._fit_hyperparameters = MultiLevel(
            {
                1: GaussianProcessHyperparameters(
                    corr_length_scales=[1.0, 1.0], process_var=1.0, nugget=0.0
                ),
                2: GaussianProcessHyperparameters(
                    corr_length_scales=[0.5, 2.0], process_var=4.0, nugget=0.1
                ),
            }
        )
        inputs1 = [Input(0, 0), Input(1, 1)]
        inputs2 = [Input(0.5, 0.5)]

        # Both inputs1 points are offset (0.5, 0.5) from inputs2[0]
        expected = np.exp(-0.5 * ((0.5 / 0.5) ** 2 + (0.5 / 2.0) ** 2)) * np.ones((2, 1))

        self.assertEqualWithinTolerance(expected, self.gp.correlation(inputs1, inputs2))
        self.assertEqual(0, self.gp.correlation([], inputs2).size)
        self.assertEqual(0, BayHEMGP().correlation(inputs1, inputs2).size)

    def test_kinv_and_covariance_matrix_not_implemented(self):
        """kinv and covariance_matrix are not supported (see issue #452)."""

        with self.assertRaises(NotImplementedError):
            self.gp.kinv

        with self.assertRaises(NotImplementedError):
            self.gp.covariance_matrix([self.x])

    @staticmethod
    def _closed_form_posterior(x, X, y, hp, beta):
        """Textbook GP posterior at `x` for a constant mean `beta`, a squared-exponential
        kernel and a nugget standard deviation, using hyperparameters `hp`."""

        ls = np.asarray(hp.corr_length_scales, dtype=float)

        def k(A, B):
            return hp.process_var * np.exp(
                -0.5 * (((A[:, None, :] - B[None, :, :]) / ls) ** 2).sum(-1)
            )

        K = k(X, X) + np.eye(len(X)) * hp.nugget**2
        kx = k(X, x)
        w = np.linalg.solve(K, kx)
        return (beta + w.T @ (y - beta)).item(), (k(x, x) - kx.T @ w).item()

    def test_fit_MAP_single_level_predict_matches_closed_form(self):
        """A MAP fit on a single level predicts exactly the closed-form GP posterior
        for the fitted hyperparameters."""

        X = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 0.1]])
        y = np.array([1.0, 2.0, 3.1, 4.0, 2.5])
        gp = BayHEMGP()
        gp.fit(MultiLevel({1: TrainingDatum.list_from_arrays(X, y)}), MAP=True)

        self.assertEqual({1}, set(gp.fit_hyperparameters.levels))
        x = np.array([[0.25, 0.35]])
        mean, var = self._closed_form_posterior(
            x, X, y, gp.fit_hyperparameters[1], float(gp._MAP["beta_L1"])
        )

        prediction = gp.predict(Input(0.25, 0.35))
        self.assertIsInstance(prediction, GaussianProcessPrediction)
        self.assertEqualWithinTolerance(mean, prediction.estimate)
        self.assertEqualWithinTolerance(var, prediction.variance)

    def test_fit_MAP_two_levels_and_predict(self):
        """A MAP fit on two levels gives finite predictions whose uncertainty is smaller
        at a top-level training input than far away from all the data."""

        X1 = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 0.1]])
        X2 = np.array([[0.2, 0.3], [0.6, 0.7]])
        training_data = MultiLevel(
            {
                1: TrainingDatum.list_from_arrays(X1, X1.sum(axis=1)),
                2: TrainingDatum.list_from_arrays(X2, X2.sum(axis=1) + 0.5),
            }
        )
        gp = BayHEMGP()
        gp.fit(training_data, MAP=True)

        self.assertEqual({1, 2}, set(gp.fit_hyperparameters.levels))
        near = gp.predict(Input(0.2, 0.3))
        far = gp.predict(Input(5.0, 5.0))
        for prediction in (near, far):
            self.assertIsInstance(prediction, GaussianProcessPrediction)
            self.assertTrue(np.isfinite(prediction.estimate))
            self.assertGreaterEqual(prediction.variance, 0)

        self.assertLess(near.variance, far.variance)

    def test_fit_sampling_and_predict(self):
        """A full MCMC fit (a handful of draws) clears any earlier MAP fit, records the
        trace and per-level hyperparameters, and supports prediction."""

        X = np.array([[0.1, 0.2], [0.4, 0.5], [0.8, 0.7]])
        training_data = MultiLevel({1: TrainingDatum.list_from_arrays(X, X.sum(axis=1))})
        gp = BayHEMGP()
        gp.fit(training_data, MAP=True)
        gp.fit(
            training_data,
            draws=10,
            tune=10,
            sample_kwargs={
                "chains": 1,
                "cores": 1,
                "progressbar": False,
                "random_seed": 1,
            },
        )

        self.assertIsNone(gp._MAP)
        self.assertIsNotNone(gp._trace)
        self.assertEqual({1}, set(gp.fit_hyperparameters.levels))
        prediction = gp.predict(Input(0.3, 0.3))
        self.assertTrue(np.isfinite(prediction.estimate))
        self.assertGreaterEqual(prediction.variance, 0)
