import unittest
from exauq.core.modelling import (
    Input,
    TrainingDatum
    )
import numpy as np


class TestInput(unittest.TestCase):
    def test_input_reals(self):
        """Test that an input can be constructed from args that are
        real numbers."""
        _ = Input(np.float16(1.2345))
        _ = Input(1.2, np.int32(33))
    
    def test_input_non_real_error(self):
        """Test that TypeError is raised during construction if there is an
        arg that doesn't define a real number."""
        with self.assertRaises(TypeError) as cm:
            Input(1, 'a')
        
        self.assertEqual('Arguments must be instances of real numbers',
                         str(cm.exception))
        
        with self.assertRaises(TypeError) as cm:
            Input(1, complex(1, 1))
        
        self.assertEqual('Arguments must be instances of real numbers',
                         str(cm.exception))

    def test_input_none_error(self):
        """Test that a TypeError is raised if the input array contains None."""

        with self.assertRaises(TypeError) as cm:
            _ = Input(1.1, None)
        
        self.assertEqual(
            "Cannot supply None as an argument", str(cm.exception)
        )

    def test_input_non_finite_error(self):
        """Test that a ValueError is raised if the input array contains various
        non-real elements."""

        msg = "Cannot supply NaN or non-finite numbers as arguments"
        with self.assertRaises(ValueError) as cm:
            _ = Input(1.1, np.nan)
        
        self.assertEqual(msg, str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            _ = Input(1.1, np.inf)
        
        self.assertEqual(msg, str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            _ = Input(1.1, np.NINF)  # negative inf
        
        self.assertEqual(msg, str(cm.exception))

    def test_str(self):
        """Test that the string description of an instance of
        Input gives the coordinates."""

        self.assertEqual("(1, 2, 3)", str(Input(1, 2, 3)))
        self.assertEqual("1.5", str(Input(1.5)))
        self.assertEqual("()", str(Input()))

    def test_repr(self):
        """Test that the string representation of an instance of
        Input gives a recipe for construction."""

        self.assertEqual("Input(1, 2, 3)", repr(Input(1, 2, 3)))
        self.assertEqual("Input(1.5)", repr(Input(1.5)))
        self.assertEqual("Input()", repr(Input()))

    def test_eq(self):
        """Test equality and inequality of different Input objects."""
        self.assertEqual(Input(), Input())
        self.assertEqual(Input(2), Input(2))
        self.assertEqual(Input(2, 3, 4), Input(2, 3, 4))
        self.assertNotEqual(Input(1), 1)
        self.assertNotEqual(Input(), Input(1))
        self.assertNotEqual(Input(2), Input(3))
        self.assertNotEqual(Input(1), Input(1, 1))
        self.assertNotEqual(Input(1, 1), Input(1, 1.1))

    def test_value_none(self):
        """Test that the value of an input is None when the input
        has no coordinates."""
        self.assertIsNone(Input().value)

    def test_value_float(self):
        """Test that the value of an input is returned as a float
        when the input has only one dimension."""
        x = Input(2)
        self.assertEqual(2, x.value)
    
    def test_value_tuple(self):
        """Test that the value of an input is a tuple when the
        input has dimension greater than one."""
        x = Input(1, 2)
        self.assertEqual((1, 2), x.value)
    
    def test_value_immutable(self):
        """Test that the value of the input is immutable."""
        x = Input(2)
        with self.assertRaises(AttributeError):
            x.value = 3

    def test_from_array(self):
        """Test that an input can be created from a Numpy array of data."""

        _input = np.array([1, -2.1, 3e2])
        expected = Input(1, -2.1, 3e2)
        self.assertEqual(expected, Input.from_array(_input))
    
    def test_from_array_not_array_error(self):
        """Test that a TypeError is raised if the input is not a Numpy array."""

        with self.assertRaises(TypeError) as cm:
            _ = Input.from_array(1)
        
        self.assertEqual(
            "'input' must be a Numpy ndarray", str(cm.exception)
            )

    def test_from_array_wrong_shape_error(self):
        """Test that a ValueError is raised if the input array is not
        1-dimensional."""

        with self.assertRaises(ValueError) as cm:
            _ = Input.from_array(np.array([[1.1], [2.2]]))
        
        self.assertEqual(
            "'input' must be a 1-dimensional Numpy array", str(cm.exception)
            )

    def test_from_array_non_real_error(self):
        """Test that a ValueError is raised if the input array contains data
        that does not define a real number."""

        with self.assertRaises(ValueError) as cm:
            _ = Input.from_array(np.array([np.datetime64(123, 'm')]))
        
        self.assertEqual(
            "'input' must be a Numpy array of real numbers", str(cm.exception)
        )
    
    def test_from_array_none_error(self):
        """Test that a ValueError is raised if the input array contains None."""

        with self.assertRaises(ValueError) as cm:
            _ = Input.from_array(np.array([1.1, None]))
        
        self.assertEqual(
            "'input' cannot contain None", str(cm.exception)
        )

    def test_from_array_not_finite_error(self):
        """Test that a ValueError is raised if the input array contains NaN or
        non-finite elements."""

        msg = "'input' cannot contain NaN or non-finite numbers"

        with self.assertRaises(ValueError) as cm:
            _ = Input.from_array(np.array([1.1, np.nan]))
        
        self.assertEqual(msg, str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            _ = Input.from_array(np.array([1.1, np.inf]))
        
        self.assertEqual(msg, str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            _ = Input.from_array(np.array([1.1, np.NINF]))  # negative inf
        
        self.assertEqual(msg, str(cm.exception))


class TestTrainingDatum(unittest.TestCase):
    def test_input_error(self):
        """Test that a TypeError is raised if the constructor arg `input`
        is not an Input."""
        with self.assertRaises(TypeError) as cm:
            TrainingDatum(1, 1)
        
        self.assertEqual('Argument `input` must be of type Input',
                         str(cm.exception))

    def test_output_error(self):
        """Test that a TypeError is raised if the constructor arg `output`
        is not an real number."""
        with self.assertRaises(TypeError) as cm:
            TrainingDatum(Input(1), 'a')
        
        self.assertEqual('Argument `output` must define a real number',
                         str(cm.exception))

        with self.assertRaises(TypeError) as cm:
            TrainingDatum(Input(1), complex(1, 1))
        
        self.assertEqual('Argument `output` must define a real number',
                         str(cm.exception))
    
    def test_str(self):
        """Test that the string description of an instance of
        TrainingDatum gives a tuple of the constituent parts."""
        
        self.assertEqual("((1, 2), 3)", str(TrainingDatum(Input(1, 2), 3)))

    def test_repr(self):
        """Test that the string representation of an instance of TrainingDatum
        gives a recipe for its construction."""

        expected = "TrainingDatum(input=Input(1, 2), output=3)"
        self.assertEqual(expected, repr(TrainingDatum(Input(1, 2), 3)))

    def test_immutable(self):
        """Test that the input and output attributes are immutable."""

        datum = TrainingDatum(Input(1), 2)
        with self.assertRaises(AttributeError) as cm:
            datum.input = Input(2)
        
        self.assertTrue(str(cm.exception).endswith("cannot assign to field 'input'"))
        
        with self.assertRaises(AttributeError) as cm:
            datum.output = 1

        self.assertTrue(str(cm.exception).endswith("cannot assign to field 'output'"))

    def test_list_from_arrays(self):
        """Test that a list of training data is created from Numpy arrays of
        inputs and outputs."""

        inputs = np.array([[0.2, 1.1], [-2, 3000.9], [3.5, 9.87]])
        outputs = np.array([-1, 0, 1.1])
        expected = [TrainingDatum(Input(0.2, 1.1), -1),
                    TrainingDatum(Input(-2, 3000.9), 0),
                    TrainingDatum(Input(3.5, 9.87), 1.1)]
        self.assertEqual(expected, TrainingDatum.list_from_arrays(inputs, outputs))
        

if __name__ == "__main__":
    unittest.main()
