import unittest
import tests.unit.fakes as fakes
from exauq.core.designers import SingleLevelAdaptiveSampler
from exauq.core.modelling import (
    Experiment,
    TrainingDatum
    )


class TestSingleLevelAdaptiveSampler(unittest.TestCase):
    initial_design = [Experiment(0.2), Experiment(0.55)]
    
    def test_str(self):
        """Test that the string description of an instance of
        SingleLevelAdaptiveSampler designer is derived from its constituent
        parts."""

        designer = SingleLevelAdaptiveSampler(self.initial_design)

        expected = f"SingleLevelAdaptiveSampler designer with initial design {str(self.initial_design)}"
        self.assertEqual(expected, str(designer))
    
    def test_repr(self):
        """Test that the string representation of an instance of
        SingleLevelAdaptiveSampler designer is derived from its constituent
        parts."""

        designer = SingleLevelAdaptiveSampler(self.initial_design)

        expected = f"SingleLevelAdaptiveSampler(initial_design={repr(self.initial_design)})"
        self.assertEqual(expected, repr(designer))
    
    def test_train(self):
        """Test that training an emulator with the SLAS designer returns an
        emulator of the same type."""
        
        designer = SingleLevelAdaptiveSampler(self.initial_design)
        simulator = fakes.OneDimSimulator(0, 1)
        emulator = fakes.DumbEmulator()
        
        trained_emulator = designer.train(emulator, simulator)
        
        self.assertIsInstance(trained_emulator, type(emulator))
    
    def test_train_fits_with_initial_design(self):
        """Test that the emulator returned by the SLAS designer has been trained
        on an initial design."""

        simulator = fakes.OneDimSimulator(0, 1)
        emulator = fakes.DumbEmulator()
        initial_design = [Experiment(0.2), Experiment(0.55)]
        
        designer = SingleLevelAdaptiveSampler(initial_design)

        trained_emulator = designer.train(emulator, simulator)
        expected = [TrainingDatum(Experiment(0.2), 0.2),
                    TrainingDatum(Experiment(0.55), 0.55)]
        self.assertEqual(expected, trained_emulator.training_data[0:2])

    def test_train_returns_new_emulator(self):
        """Test that training an emulator returns a new emulator object,
        leaving the original unchanged."""

        simulator = fakes.OneDimSimulator(0, 1)
        emulator = fakes.DumbEmulator()
        designer = SingleLevelAdaptiveSampler(self.initial_design)
        
        trained_emulator = designer.train(emulator, simulator)
        self.assertNotEqual(emulator, trained_emulator)
        self.assertIsNone(emulator.training_data)


if __name__ == "__main__":
    unittest.main()
