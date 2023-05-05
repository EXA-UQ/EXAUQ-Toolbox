import unittest
import mogp_emulator as mogp
import numpy as np
from exauq.core.emulators import MogpEmulator


class TestMogpEmulator(unittest.TestCase):
    
    def test_constructor(self):
        """Test that an instance of MogpEmulator can be constructed from an
        mogp GaussianProcess object."""
        
        inputs = np.array([[0, 0],
                           [0.2, 0.1]])
        outputs = np.array([1, 2])
        gp = mogp.GaussianProcess(inputs, outputs)
        _ = MogpEmulator(gp)
    
    def test_constructor_error(self):
        """Test that a ValueError is raised if an argument of type different
        to mogp's GaussianProcess class is passed to the constructor.
        """

        with self.assertRaises(TypeError) as cm:
            MogpEmulator(1)
        
        expected_msg = ("Argument 'gp' must be of type GaussianProcess "
                        "from the mogp-emulator package")
        self.assertEqual(expected_msg, str(cm.exception))


if __name__ == "__main__":
    unittest.main()
