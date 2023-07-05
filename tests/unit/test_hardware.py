import unittest

from exauq.core.hardware import HardwareInterface


class TestHardwareInterface(unittest.TestCase):
    # Test that an error is raised when trying to instantiate the abstract base class
    def test_abc_instantiation(self):
        with self.assertRaises(TypeError):
            a = HardwareInterface()


if __name__ == "__main__":
    unittest.main()
