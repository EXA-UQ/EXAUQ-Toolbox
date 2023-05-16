import unittest
import copy
import mogp_emulator as mogp
import numpy as np
from exauq.core.emulators import MogpEmulator
from exauq.core.modelling import (
    Input,
    TrainingDatum
)


class TestMogpEmulator(unittest.TestCase):
    gp = mogp.GaussianProcess(np.array([[0, 0], [0.2, 0.1]]), np.array([1, 2]))

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

        # Note: need to use allclose because fitting it not deterministic.
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

    def test_fit_no_training_data_with_bounds(self):
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
        corr = gp2.theta.corr_raw
        cov = gp2.theta.get_data()[-1]
        bounds = (
            (0.8 * corr[0], 0.9 * corr[0]),
            (0.8 * corr[1], 0.9 * corr[1]),
            (1.1 * cov, 1.2 * cov)
            )

        emulator = MogpEmulator(gp)
        emulator.fit(hyperparameter_bounds=bounds)
        actual_corr = emulator.gp.theta.corr_raw
        actual_cov = emulator.gp.theta.get_data()[-1]
        
        self.assertTrue(
            bounds[0][0] <= actual_corr[0] <= bounds[0][1]
        )
        self.assertTrue(
            bounds[1][0] <= actual_corr[1] <= bounds[1][1]
        )
        self.assertTrue(
            bounds[2][0] <= actual_cov <= bounds[2][1]
        )


if __name__ == "__main__":
    unittest.main()
