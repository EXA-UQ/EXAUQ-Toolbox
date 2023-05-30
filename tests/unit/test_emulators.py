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
    gp = mogp.GaussianProcess(np.array([[0, 0], [0.2, 0.1]]), np.array([1, 2]))

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
        assert (
            almost_between,
            "'x' not between lower and upper bounds to given tolerance"
            )

    def test_constructor(self):
        """Test that an instance of MogpEmulator can be constructed from an
        mogp GaussianProcess object."""
        
        _ = MogpEmulator(self.gp)
    
    def test_constructor_error(self):
        """Test that a ValueError is raised if an argument of type different
        to mogp's GaussianProcess class is passed to the constructor.
        """

        with self.assertRaises(TypeError) as cm:
            MogpEmulator(1)
        
        expected_msg = ("Argument 'gp' must be of type GaussianProcess "
                        "from the mogp-emulator package")
        self.assertEqual(expected_msg, str(cm.exception))
    
    def test_gp_is_identical(self):
        """Test that the underlying GP is the GaussianProcess
        object supplied upon the emulator's construction."""

        emulator = MogpEmulator(self.gp)
        self.assertEqual(id(self.gp), id(emulator.gp))

    def test_gp_immutable(self):
        """Test that the underlying GP cannot be directly modified."""
        
        emulator = MogpEmulator(self.gp)
        with self.assertRaises(AttributeError):
            emulator.gp = mogp.GaussianProcess(np.array([[0.5], [0.3]]), np.array([1, 2]))

    def test_training_data(self):
        """Test that the training data in the underlying GP can be recovered
        before any further training has taken place."""

        gp = mogp.GaussianProcess(np.array([[0.5], [0.3]]), np.array([1, 2]))
        emulator = MogpEmulator(gp)
        expected = [TrainingDatum(Input(0.5), 1), TrainingDatum(Input(0.3), 2)]
        self.assertEqual(expected, emulator.training_data)

    def test_fit_no_training_data(self):
        """Test that fitting the emulator with no training data supplied results
        in the underlying GP being fit with hyperparameter estimation."""

        gp1 = mogp.GaussianProcess(
            inputs=np.array([[0, 0],
                             [0.2, 0.1],
                             [0.3, 0.5],
                             [0.7, 0.4],
                             [0.9, 0.8]]),
            targets=np.array([1, 2, 3.1, 9, 2])
            )
        gp2 = copy.deepcopy(gp1)
        gp1 = mogp.fit_GP_MAP(gp1)
        emulator = MogpEmulator(gp2)
        emulator.fit()

        # Note: need to use allclose because fitting is not deterministic.
        self.assertTrue(
            np.allclose(
                gp1.theta.corr, emulator.gp.theta.corr, rtol=1e-5, atol = 0
            )
        )
        self.assertTrue(
            np.allclose(
                gp1.theta.cov, emulator.gp.theta.cov, rtol=1e-5, atol = 0
            )
        )
        self.assertTrue(
            np.allclose(
                gp1.theta.nugget, emulator.gp.theta.nugget, rtol=1e-5, atol = 0
            )
        )

    def test_fit_with_bounds(self):
        """Test that fitting the emulator respects bounds on hyperparameters
        when these are supplied."""

        gp = mogp.GaussianProcess(
            inputs=np.array([[0, 0],
                             [0.2, 0.1],
                             [0.3, 0.5],
                             [0.7, 0.4],
                             [0.9, 0.8]]),
            targets=np.array([1, 2, 3.1, 9, 2])
            )
        gp2 = copy.deepcopy(gp)
        gp2 = mogp.fit_GP_MAP(gp2)

        # Compute bounds to apply, by creating small windows away from known
        # optimal values of the hyperparameters.
        corr = gp2.theta.corr
        cov = gp2.theta.cov
        bounds = (
            (0.8 * corr[0], 0.9 * corr[0]),
            (0.8 * corr[1], 0.9 * corr[1]),
            (1.1 * cov, 1.2 * cov)
            )

        emulator = MogpEmulator(gp)
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


if __name__ == "__main__":
    unittest.main()
