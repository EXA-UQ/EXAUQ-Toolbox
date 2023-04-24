import unittest
from exauq.core.modelling import (
    Experiment,
    TrainingDatum
    )


class TestExperiment(unittest.TestCase):
    def test_eq(self):
        """Test equality and inequality of different Experiment objects."""
        self.assertEqual(Experiment(), Experiment())
        self.assertEqual(Experiment(2), Experiment(2))
        self.assertEqual(Experiment(2, 3, 4), Experiment(2, 3, 4))
        self.assertNotEqual(Experiment(1), 1)
        self.assertNotEqual(Experiment(), Experiment(1))
        self.assertNotEqual(Experiment(2), Experiment(3))
        self.assertNotEqual(Experiment(1), Experiment(1, 1))
        self.assertNotEqual(Experiment(1, 1), Experiment(1, 1.1))

    def test_value_none(self):
        """Test that the value of an experiment is None when the experiment
        has no coordinates."""
        self.assertIsNone(Experiment().value)

    def test_value_float(self):
        """Test that the value of an experiment is returned as a float
        when the experiment has only one dimension."""
        x = Experiment(2)
        self.assertEqual(2, x.value)
    
    def test_value_tuple(self):
        """Test that the value of an experiment is a tuple when the
        experiment has dimension greater than one."""
        x = Experiment(1, 2)
        self.assertEqual((1, 2), x.value)
    
    def test_value_immutable(self):
        """Test that the value of the experiment is immutable."""
        x = Experiment(2)
        with self.assertRaises(AttributeError):
            x.value = 3


class TestTrainingDatum(unittest.TestCase):
    def test_immutable(self):
        """Test that the experiment and observation attributes are immutable."""

        datum = TrainingDatum(Experiment(1), 2)
        with self.assertRaises(AttributeError):
            datum.experiment = Experiment(2)
        
        with self.assertRaises(AttributeError):
            datum.observation = 1


if __name__ == "__main__":
    unittest.main()
