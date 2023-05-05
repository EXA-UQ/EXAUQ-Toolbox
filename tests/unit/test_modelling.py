import unittest
from exauq.core.modelling import (
    Experiment,
    TrainingDatum
    )
import numpy as np


class TestExperiment(unittest.TestCase):
    def test_input_reals(self):
        """Test that an experiment can be constructed from args that are
        real numbers."""
        _ = Experiment(np.float16(1.2345))
        _ = Experiment(1.2, np.int32(33))
    
    def test_input_non_real_error(self):
        """Test that TypeError is raised during construction if there is an
        arg that doesn't define a real number."""
        with self.assertRaises(TypeError) as cm:
            Experiment(1, 'a')
        
        self.assertEqual('Arguments must be instances of real numbers',
                         str(cm.exception))
        
        with self.assertRaises(TypeError) as cm:
            Experiment(1, complex(1, 1))
        
        self.assertEqual('Arguments must be instances of real numbers',
                         str(cm.exception))

    def test_str(self):
        """Test that the string description of an instance of
        Experiment gives the coordinates."""

        self.assertEqual("(1, 2, 3)", str(Experiment(1, 2, 3)))
        self.assertEqual("1.5", str(Experiment(1.5)))
        self.assertEqual("()", str(Experiment()))

    def test_repr(self):
        """Test that the string representation of an instance of
        Experiment gives a recipe for construction."""

        self.assertEqual("Experiment(1, 2, 3)", repr(Experiment(1, 2, 3)))
        self.assertEqual("Experiment(1.5)", repr(Experiment(1.5)))
        self.assertEqual("Experiment()", repr(Experiment()))

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
    def test_input_experiment_error(self):
        """Test that a TypeError is raised if the constructor arg `experiment`
        is not an Experiment."""
        with self.assertRaises(TypeError) as cm:
            TrainingDatum(1, 1)
        
        self.assertEqual('Argument `experiment` must be of type Experiment',
                         str(cm.exception))

    def test_input_observation_error(self):
        """Test that a TypeError is raised if the constructor arg `observation`
        is not an real number."""
        with self.assertRaises(TypeError) as cm:
            TrainingDatum(Experiment(1), 'a')
        
        self.assertEqual('Argument `observation` must define a real number',
                         str(cm.exception))

        with self.assertRaises(TypeError) as cm:
            TrainingDatum(Experiment(1), complex(1, 1))
        
        self.assertEqual('Argument `observation` must define a real number',
                         str(cm.exception))
    
    def test_str(self):
        """Test that the string description of an instance of
        TrainingDatum gives a tuple of the constituent parts."""
        
        self.assertEqual("((1, 2), 3)", str(TrainingDatum(Experiment(1, 2), 3)))

    def test_repr(self):
        """Test that the string representation of an instance of TrainingDatum
        gives a recipe for its construction."""

        expected = "TrainingDatum(experiment=Experiment(1, 2), observation=3)"
        self.assertEqual(expected, repr(TrainingDatum(Experiment(1, 2), 3)))

    def test_immutable(self):
        """Test that the experiment and observation attributes are immutable."""

        datum = TrainingDatum(Experiment(1), 2)
        with self.assertRaises(AttributeError):
            datum.experiment = Experiment(2)
        
        with self.assertRaises(AttributeError):
            datum.observation = 1


if __name__ == "__main__":
    unittest.main()
