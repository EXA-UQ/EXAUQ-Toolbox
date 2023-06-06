import unittest
import copy
import math
import mogp_emulator as mogp
import numpy as np
from exauq.core.emulators import MogpEmulator
from exauq.core.modelling import (
    Input,
    TrainingDatum
)


class TestMogpEmulator(unittest.TestCase):
    # Some default args to use for constructing mogp GaussianProcess objects
    inputs = np.array([[0, 0], [0.2, 0.1]])
    targets = np.array([1, 2])

    def assertAlmostBetween(self, x, lower, upper, **kwargs) -> None:
        """Checks whether a number lies between bounds up to a tolerance.
        
        An AssertionError is thrown unless `lower` < `x` < `upper` or `x` is
        close to either bound. Keyword arguments are passed to the function
        `math.isclose`, with `rel_tol` and `abs_tol` being used to define the
        tolerances in the closeness check (see the documentation for
        `math.isclose` on their definition and default values).
        """
        almost_between = (
            (lower < x < upper) or
            math.isclose(x, lower, **kwargs) or
            math.isclose(x, upper, **kwargs)
            )
        assert almost_between,\
            "'x' not between lower and upper bounds to given tolerance"

    def test_constructor(self):
        """Test that an instance of MogpEmulator can be constructed from args
        and kwargs required of an mogp GaussianProcess object."""
        
        # kwargs used for constructing a mogp GaussianProcess object
        mean = None
        kernel = 'Matern52'
        priors = None
        nugget = 'pivot'
        inputdict = {}
        use_patsy = False

        _ = MogpEmulator(
            self.inputs, self.targets, mean=mean, kernel=kernel, priors=priors,
            nugget=nugget, inputdict=inputdict, use_patsy=use_patsy
            )
    
    def test_constructor_error(self):
        """Test that a RuntimeError is raised if a mogp GaussianProcess
        couldn't be constructed. 
        """

        with self.assertRaises(RuntimeError) as cm:
            MogpEmulator(self.inputs, self.targets, men=None)
        
        expected_msg = ("Could not construct an underlying mogp-emulator "
                        "GaussianProcess during initialisation")
        self.assertEqual(expected_msg, str(cm.exception))
    
    def test_underlying_gp_construction(self):
        """Test that the underlying mogp GaussianProcess was constructed with
        args and kwargs supplied to the MogpEmulator initialiser."""

        # Non-default choices for some kwargs
        kernel = 'Matern52'
        nugget = 'pivot'
        emulator = MogpEmulator(
            self.inputs, self.targets, kernel=kernel, nugget=nugget
            )
        
        self.assertTrue((self.inputs == emulator.gp.inputs).all())
        self.assertTrue((self.targets == emulator.gp.targets).all())
        self.assertEqual("Matern 5/2 Kernel", str(emulator.gp.kernel))
        self.assertEqual(nugget, emulator.gp.nugget_type)

    def test_training_data(self):
        """Test that the training data in the underlying GP can be recovered
        before any further training has taken place."""

        emulator = MogpEmulator(np.array([[0.5], [0.3]]), np.array([1, 2]))
        expected = [TrainingDatum(Input(0.5), 1), TrainingDatum(Input(0.3), 2)]
        self.assertEqual(expected, emulator.training_data)

    def test_fit_no_training_data(self):
        """Test that fitting the emulator with no training data supplied results
        in the underlying GP being fit with hyperparameter estimation."""

        inputs = np.array([[0, 0],
                           [0.2, 0.1],
                           [0.3, 0.5],
                           [0.7, 0.4],
                           [0.9, 0.8]])
        targets = np.array([1, 2, 3.1, 9, 2])
        gp = mogp.fit_GP_MAP(mogp.GaussianProcess(inputs, targets))
        emulator = MogpEmulator(inputs, targets)
        emulator.fit()

        # Note: need to use allclose because fitting is not deterministic.
        self.assertTrue(
            np.allclose(
                gp.theta.corr, emulator.gp.theta.corr, rtol=1e-5, atol = 0
            )
        )
        self.assertTrue(
            np.allclose(
                gp.theta.cov, emulator.gp.theta.cov, rtol=1e-5, atol = 0
            )
        )
        self.assertTrue(
            np.allclose(
                gp.theta.nugget, emulator.gp.theta.nugget, rtol=1e-5, atol = 0
            )
        )

    def test_fit_with_bounds(self):
        """Test that fitting the emulator respects bounds on hyperparameters
        when these are supplied."""

        inputs = np.array([[0, 0],
                           [0.2, 0.1],
                           [0.3, 0.5],
                           [0.7, 0.4],
                           [0.9, 0.8]])
        targets = np.array([1, 2, 3.1, 9, 2])
        gp = mogp.fit_GP_MAP(mogp.GaussianProcess(inputs, targets))

        # Compute bounds to apply, by creating small windows away from known
        # optimal values of the hyperparameters.
        corr = gp.theta.corr
        cov = gp.theta.cov
        bounds = (
            (0.8 * corr[0], 0.9 * corr[0]),
            (0.8 * corr[1], 0.9 * corr[1]),
            (1.1 * cov, 1.2 * cov)
            )

        emulator = MogpEmulator(inputs, targets)
        emulator.fit(hyperparameter_bounds=bounds)
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
        self.assertAlmostBetween(
            actual_cov, bounds[2][0], bounds[2][1], rel_tol=1e-10
            )

    def test_compute_raw_param_bounds(self):
        """Test that correlation length parameters and the covariance are
        transformed to raw parameters. For the transformations, see:
        https://mogp-emulator.readthedocs.io/en/latest/implementation/GPParams.html#mogp_emulator.GPParams.GPParams
        """

        bounds = (
            (1, math.e),  # correlation
            (1, math.e)  # covariance
            )
        raw_bounds = MogpEmulator._compute_raw_param_bounds(bounds)
        self.assertAlmostEqual(-2, raw_bounds[0][0])
        self.assertAlmostEqual(0, raw_bounds[0][1])
        self.assertAlmostEqual(0, raw_bounds[1][0])
        self.assertAlmostEqual(1, raw_bounds[1][1])

    def test_compute_raw_param_bounds_multiple_corr(self):
        """Test that correlation length parameters and the covariance are
        transformed to raw parameters when there are multiple raw correlation
        parameters."""
        
        bounds = (
            (1, math.e),  # correlation
            (math.exp(-2), math.exp(3)),  # correlation
            (1, math.e)  # covariance
            )
        raw_bounds = MogpEmulator._compute_raw_param_bounds(bounds)
        self.assertAlmostEqual(-2, raw_bounds[0][0])
        self.assertAlmostEqual(0, raw_bounds[0][1])
        self.assertAlmostEqual(-6, raw_bounds[1][0])
        self.assertAlmostEqual(4, raw_bounds[1][1])
        self.assertAlmostEqual(0, raw_bounds[2][0])
        self.assertAlmostEqual(1, raw_bounds[2][1])

    def test_fit_training_data_not_none(self):
        """Test that a ValueError is raised if no training data is supplied
        and there is no training data already stored in the emulator."""

        emulator = MogpEmulator([], [])
        with self.assertRaises(ValueError) as cm1:
            emulator.fit()
        
        with self.assertRaises(ValueError) as cm2:
            emulator.fit(training_data=[])
        
        expected_msg = ("Cannot fit emulator if no training data supplied and "
                        "the 'training_data' property is empty")
        self.assertEqual(expected_msg, str(cm1.exception))
        self.assertEqual(expected_msg, str(cm2.exception))

    def test_fit_supplied_training_data_no_existing_training_data(self):
        """Test that the emulator is trained on the supplied training data
        and its training_data property is updated, in the case where there is
        no pre-exisiting training data."""

        inputs = np.array([[0, 0],
                           [0.2, 0.1],
                           [0.3, 0.5],
                           [0.7, 0.4],
                           [0.9, 0.8]])
        targets = np.array([1, 2, 3.1, 9, 2])
        emulator = MogpEmulator([], [])
        training_data = TrainingDatum.list_from_arrays(inputs, targets)
        emulator.fit(training_data=training_data)
        
        self.assertTrue(np.allclose(inputs, emulator.gp.inputs))
        self.assertTrue(np.allclose(targets, emulator.gp.targets))
        self.assertEqual(training_data, emulator.training_data)
    
    def test_fit_supplied_training_data_gp_kwargs(self):
        """Test that, after fitting with supplied training data, the underlying
        mogp emulator has the same settings as the original did before fitting."""

        inputs = np.array([[0, 0],
                           [0.2, 0.1],
                           [0.3, 0.5],
                           [0.7, 0.4],
                           [0.9, 0.8]])
        targets = np.array([1, 2, 3.1, 9, 2])
        emulator = MogpEmulator([], [], nugget='pivot')
        training_data = TrainingDatum.list_from_arrays(inputs, targets)
        emulator.fit(training_data=training_data)
        
        self.assertEqual('pivot', emulator.gp.nugget_type)


if __name__ == "__main__":
    unittest.main()
