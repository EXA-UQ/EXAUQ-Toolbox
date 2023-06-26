import unittest

from exauq.core.hardware import HardwareInterface


class DummyHardwareInterface(HardwareInterface):
    pass


class TestHardwareInterface(unittest.TestCase):
    def test_abc_instantiation(self):
        with self.assertRaises(TypeError):
            a = HardwareInterface(hostname="local",
                                  username="DiscoPotato",
                                  password="GroovySpuds777!")

    def test_dummy_subclass_instantiation(self):
        with self.assertRaises(TypeError):
            b = DummyHardwareInterface(hostname="local",
                                       username="SpudtasticWizard",
                                       password="MagicalPotatoes42#")
