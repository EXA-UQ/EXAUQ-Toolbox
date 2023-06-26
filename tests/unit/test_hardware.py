import unittest

from exauq.core.hardware import HardwareInterface


# Define a dummy subclass that does not implement the abstract method
class DummyHardwareInterface(HardwareInterface):
    pass


class TestHardwareInterface(unittest.TestCase):
    # Test that an error is raised when trying to instantiate the abstract base class
    def test_abc_instantiation(self):
        with self.assertRaises(TypeError):
            a = HardwareInterface(
                hostname="local", username="DiscoPotato", password="GroovySpuds777!"
            )

    # Test that an error is raised when trying to instantiate a subclass that doesn't implement
    # all abstract methods
    def test_dummy_subclass_instantiation(self):
        with self.assertRaises(TypeError):
            b = DummyHardwareInterface(
                hostname="local",
                username="SpudtasticWizard",
                password="MagicalPotatoes42#",
            )


if __name__ == "__main__":
    unittest.main()
