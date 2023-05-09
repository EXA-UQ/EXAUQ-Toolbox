import unittest
import mogp_emulator as mogp
import numpy as np
from exauq.core.emulators import MogpEmulator


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


if __name__ == "__main__":
    unittest.main()
