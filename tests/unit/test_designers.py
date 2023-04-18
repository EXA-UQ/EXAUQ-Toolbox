import unittest
import tests.unit.fakes as fakes
from exauq.core.designers import SingleLevelAdaptiveSampler
from exauq.core.modelling import AbstractEmulator

class TestSingleLevelAdaptiveSampler(unittest.TestCase):
    def test_designer_str(self):
        """Test that the string description of an instance of
        SingleLevelAdaptiveSampler designer is derived from its constituent
        parts."""

        simulator = fakes.OneDimSimulator(0, 1)
        emulator = fakes.DumbEmulator()
        designer = SingleLevelAdaptiveSampler(emulator, simulator)

        expected = f"SingleLevelAdaptiveSampler designer for simulator {str(simulator)}, " \
                   f"using emulator {str(emulator)}"
        self.assertEqual(expected, str(designer))

    def test_designer_repr(self):
        """Test that the string representation of an instance of
        SingleLevelAdaptiveSampler designer is derived from its constituent
        parts."""

        simulator = fakes.OneDimSimulator(0, 1)
        emulator = fakes.DumbEmulator()
        designer = SingleLevelAdaptiveSampler(emulator, simulator)

        expected = f"SingleLevelAdaptiveSampler(simulator={repr(simulator)}, " \
                   f"emulator={repr(emulator)})"
        self.assertEqual(expected, repr(designer))

    def test_run(self):
        """Test that running the SLAS designer returns an emulator of the
        same type as supplied to it."""
        
        simulator = fakes.OneDimSimulator(0, 1)
        emulator = fakes.DumbEmulator()
        designer = SingleLevelAdaptiveSampler(emulator, simulator)
        
        trained_emulator = designer.run()
        
        self.assertIsInstance(trained_emulator, type(emulator))
    
    def test_run_returns_trained_emulator(self):
        """Test that running the SLAS designer returns an emulator that contains
        training data."""

        simulator = fakes.OneDimSimulator(0, 1)
        emulator = fakes.DumbEmulator()
        designer = SingleLevelAdaptiveSampler(emulator, simulator)
        
        trained_emulator = designer.run()

        self.assertIsNotNone(trained_emulator.training_data)

    def test_run_returns_new_emulator(self):
        """Test that running the SLAS designer returns a new emulator object,
        leaving the original unchanged."""

        simulator = fakes.OneDimSimulator(0, 1)
        emulator = fakes.DumbEmulator()
        designer = SingleLevelAdaptiveSampler(emulator, simulator)
        
        trained_emulator = designer.run()
        self.assertNotEqual(emulator, trained_emulator)
        self.assertIsNone(emulator.training_data)


if __name__ == "__main__":
    unittest.main()
