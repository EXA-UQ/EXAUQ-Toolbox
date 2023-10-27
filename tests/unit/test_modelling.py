import itertools
import unittest

import numpy as np

from exauq.core.modelling import Input, Prediction, SimulatorDomain, TrainingDatum
from exauq.core.numerics import FLOAT_TOLERANCE, equal_within_tolerance
from tests.utilities.utilities import exact, make_window


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
            Input(1, "a")

        self.assertEqual("Arguments must be instances of real numbers", str(cm.exception))

        with self.assertRaises(TypeError) as cm:
            Input(1, complex(1, 1))

        self.assertEqual("Arguments must be instances of real numbers", str(cm.exception))

    def test_input_none_error(self):
        """Test that a TypeError is raised if the input array contains None."""

        with self.assertRaises(TypeError) as cm:
            _ = Input(1.1, None)

        self.assertEqual(
            "Input coordinates must be real numbers, not None", str(cm.exception)
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

    def test_len(self):
        """Test that the length of the input is equal to the number of coordinates (or
        0 if empty)."""
        self.assertEqual(0, len(Input()))
        self.assertEqual(1, len(Input(0)))
        self.assertEqual(2, len(Input(0, 0)))

    def test_getitem_int(self):
        """Test that individual coordinates of the input can be accessed with ints
        (with indexing starting at 0)."""

        self.assertEqual(1, Input(1)[0])
        self.assertEqual(2, Input(1, 2)[1])
        self.assertEqual(2, Input(1, 2)[-1])

    def test_getitem_slice(self):
        """Test that a slice of the coordinates can be accessed, returned as a new Input
        instance."""

        x1 = Input(1, 2)
        x2 = Input(1, 2, 3, 4)
        x3 = Input(1)
        self.assertEqual(Input(1), x1[0:1])
        self.assertEqual(Input(1), x1[:-1])
        self.assertEqual(Input(1, 2), x1[:])
        self.assertEqual(Input(2), x1[1:])
        self.assertEqual(Input(1, 3), x2[::2])
        self.assertEqual(Input(2, 4), x2[1::2])
        self.assertEqual(Input(2), x2[1:3:2])
        self.assertEqual(Input(1), x3[:])
        self.assertEqual(Input(), x3[:0])

    def test_getitem_wrong_type_error(self):
        """Test that a TypeError is raised if the indexing item is not an integer."""

        i = "a"
        x = Input(2)
        with self.assertRaisesRegex(
            TypeError,
            exact(f"Subscript must be an 'int' or slice, but received {type(i)}."),
        ):
            x[i]

    def test_getitem_index_out_of_bounds_error(self):
        """Test that an IndexError is raised if the indexing item does not fall within
        the input's dimension."""

        x = Input(2)
        i = 1
        with self.assertRaisesRegex(IndexError, exact(f"Input index {i} out of range.")):
            x[i]

    def test_sequence_implementation(self):
        """Test that an Input implements the collections.abc.Sequence interface."""

        for method in [
            "__getitem__",
            "__len__",
            "__contains__",
            "__iter__",
            "__reversed__",
            "index",
            "count",
        ]:
            with self.subTest(method=method):
                self.assertTrue(hasattr(Input, method))

    def test_from_array(self):
        """Test that an input can be created from a Numpy array of data."""

        _input = np.array([1, -2.1, 3e2])
        expected = Input(1, -2.1, 3e2)
        self.assertEqual(expected, Input.from_array(_input))

    def test_from_array_not_array_error(self):
        """Test that a TypeError is raised if the input is not a Numpy array."""

        x = 1
        with self.assertRaises(TypeError) as cm:
            _ = Input.from_array(x)

        self.assertEqual(
            f"Expected 'input' of type numpy.ndarray but received {type(x)}.",
            str(cm.exception),
        )

    def test_from_array_wrong_shape_error(self):
        """Test that a ValueError is raised if the input array is not
        1-dimensional."""

        arr = np.array([[1.1], [2.2]])
        with self.assertRaises(ValueError) as cm:
            _ = Input.from_array(arr)

        self.assertEqual(
            "Expected 'input' to be a 1-dimensional numpy.ndarray but received an "
            f"array with {arr.ndim} dimensions.",
            str(cm.exception),
        )

    def test_from_array_non_real_error(self):
        """Test that a ValueError is raised if the input array contains data
        that does not define a real number."""

        with self.assertRaises(ValueError) as cm:
            _ = Input.from_array(np.array([np.datetime64(123, "m")]))

        self.assertEqual(
            "'input' must be a numpy.ndarray array of real numbers", str(cm.exception)
        )

    def test_from_array_none_error(self):
        """Test that a ValueError is raised if the input array contains None."""

        with self.assertRaises(ValueError) as cm:
            _ = Input.from_array(np.array([1.1, None]))

        self.assertEqual("'input' cannot contain None", str(cm.exception))

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

        self.assertEqual("Argument `input` must be of type Input", str(cm.exception))

    def test_output_not_real_error(self):
        """Test that a TypeError is raised if the constructor arg `output`
        is not a real number."""

        msg = "Argument `output` must define a real number"
        with self.assertRaises(TypeError) as cm:
            TrainingDatum(Input(1), "a")

        self.assertEqual(msg, str(cm.exception))

        with self.assertRaises(TypeError) as cm:
            TrainingDatum(Input(1), complex(1, 1))

        self.assertEqual(msg, str(cm.exception))

        with self.assertRaises(TypeError) as cm:
            TrainingDatum(Input(1), [1.1])

        self.assertEqual(msg, str(cm.exception))

    def test_output_none_error(self):
        """Test that a TypeError is raised if the constructor arg `output`
        is None."""

        with self.assertRaises(TypeError) as cm:
            _ = TrainingDatum(Input(1), None)

        self.assertEqual("Argument 'output' cannot be None", str(cm.exception))

    def test_output_not_finite_error(self):
        """Test that a ValueError is raised if the constructor arg `output` is
        NaN or non-finite."""

        msg = "Argument 'output' cannot be NaN or non-finite"

        with self.assertRaises(ValueError) as cm:
            _ = TrainingDatum(Input(1), np.nan)

        self.assertEqual(msg, str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            _ = TrainingDatum(Input(1), np.inf)

        self.assertEqual(msg, str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            _ = TrainingDatum(Input(1), np.NINF)  # negative inf

        self.assertEqual(msg, str(cm.exception))

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
        expected = [
            TrainingDatum(Input(0.2, 1.1), -1),
            TrainingDatum(Input(-2, 3000.9), 0),
            TrainingDatum(Input(3.5, 9.87), 1.1),
        ]
        self.assertEqual(expected, TrainingDatum.list_from_arrays(inputs, outputs))


class TestPrediction(unittest.TestCase):
    def test_inputs_preserve_real_type(self):
        """Test that the mean and variance are of the same types as provided
        at initialisation."""

        means = [1, 1.2, np.float16(1.3)]
        variances = [1.2, np.float16(1.3), 1]

        for mean, var in zip(means, variances):
            prediction = Prediction(mean, var)
            self.assertIsInstance(prediction.mean, type(mean))
            self.assertIsInstance(prediction.variance, type(var))

    def test_non_real_error(self):
        """Test that a TypeError is raised if the supplied mean or variance is not a
        real number."""

        non_real = "1"
        with self.assertRaisesRegex(
            TypeError,
            exact(
                f"Expected 'mean' to define a real number, but received {type(non_real)} "
                "instead."
            ),
        ):
            Prediction(mean=non_real, variance=1)

        with self.assertRaisesRegex(
            TypeError,
            exact(
                f"Expected 'variance' to define a real number, but received {type(non_real)} "
                "instead."
            ),
        ):
            Prediction(mean=1, variance=non_real)

    def test_negative_variance_error(self):
        """Test that a ValueError is raised if a negative variance is provided at
        initialisation."""

        var = -0.1
        with self.assertRaisesRegex(
            ValueError,
            exact(f"'variance' must be a non-negative real number, but received {var}."),
        ):
            Prediction(mean=1, variance=var)

    def test_immutable_fields(self):
        """Test that the mean and variance values in a prediction are immutable."""

        prediction = Prediction(1, 1)
        with self.assertRaises(AttributeError):
            prediction.mean = 0

        with self.assertRaises(AttributeError):
            prediction.variance = 0

    def test_equality_with_different_type(self):
        """Test that a Prediction object is not equal to an object of a different type."""

        p = Prediction(mean=1, variance=1)
        for other in [1, "1", (1, 1)]:
            self.assertNotEqual(p, other)

    def test_equality_of_means(self):
        """Given two predictions with the same variances, test that they are equal if
        and only if the means are close according to the default tolerance."""

        variance = 0
        for mean1 in [0.5 * n for n in range(-100, 101)]:
            p1 = Prediction(mean1, variance)
            for mean2 in make_window(mean1, tol=FLOAT_TOLERANCE):
                p2 = Prediction(mean2, variance)
                self.assertIs(p1 == p2, equal_within_tolerance(mean1, mean2))
                self.assertIs(p2 == p1, equal_within_tolerance(mean1, mean2))

    def test_equality_of_variances(self):
        """Given two predictions with the same means, test that they are equal if
        and only if the variances are close according to the default tolerance."""

        mean = 0
        for var1 in [0.1 * n for n in range(101)]:
            p1 = Prediction(mean, var1)
            for var2 in filter(lambda x: x >= 0, make_window(var1, tol=FLOAT_TOLERANCE)):
                p2 = Prediction(mean, var2)
                self.assertIs(p1 == p2, equal_within_tolerance(var1, var2))
                self.assertIs(p2 == p1, p1 == p2)

    def test_equality_symmetric(self):
        """Test that equality of predictions doesn't depend on the order of the objects
        in the comparison."""

        for mean1, var1 in itertools.product([-0.5, 0, 0.5], [0, 0.1, 0.2]):
            p1 = Prediction(mean1, var1)
            means = make_window(mean1, tol=FLOAT_TOLERANCE)
            variances = filter(lambda x: x >= 0, make_window(var1, tol=FLOAT_TOLERANCE))
            for mean2, var2 in itertools.product(means, variances):
                p2 = Prediction(mean2, var2)
                self.assertIs(p1 == p2, p2 == p1)


class TestSimulatorDomain(unittest.TestCase):
    def setUp(self) -> None:
        self.bounds = [(0, 2), (0, 1)]
        self.domain = SimulatorDomain(self.bounds)

    def test_input_not_in_domain_wrong_type(self):
        """Test that objects not of type Input are not contained in the domain."""

        self.assertFalse(1.0 in self.domain)

    def test_input_not_in_domain_wrong_dims(self):
        """Test that an Input with the wrong number of dimensions cannot belong to
        the domain."""

        x1 = Input(1)
        x2 = Input(1, 1, 1)
        self.assertFalse(x1 in self.domain)
        self.assertFalse(x2 in self.domain)

    def test_input_not_in_domain_coord_out_of_bounds_one_dim(self):
        """Test that an Input with a coordinate that lies outside the domain's bounds
        cannot belong to the domain, in the case of a 1-dim domain."""

        bounds = [(0, 1)]
        domain = SimulatorDomain(bounds)
        x1 = Input(1.1)
        x2 = Input(-0.1)
        self.assertFalse(x1 in domain)
        self.assertFalse(x2 in domain)

    def test_input_not_in_domain_coord_out_of_bounds_multi_dim(self):
        """Test that an Input with a coordinate that lies outside the domain's bounds
        cannot belong to the domain, in the case of a multi-dim domain."""

        x1 = Input(2.1, 0)
        x2 = Input(1, -0.1)
        self.assertFalse(x1 in self.domain)
        self.assertFalse(x2 in self.domain)

    def test_dim_equal_number_of_supplied_bounds(self):
        """Test that the dimension of the domain is equal to the length of the bounds
        sequence that was supplied."""

        self.assertEqual(len(self.bounds), self.domain.dim)

    def test_dim_immutable(self):
        """Test that the dim property is read-only."""

        with self.assertRaises(AttributeError):
            self.domain.dim = 1

    def test_scale_returns_input(self):
        """Test that the scale method returns a tuple of real numbers."""

        coordinates = (0.5, 0.5)
        transformed = self.domain.scale(coordinates)
        self.assertIsInstance(transformed, Input)

    def test_scale_wrong_dimension_input_error(self):
        """Test that a ValueError is raised if the wrong number of coordinates are
        present in the input arg."""

        for coordinates in [(1,), (1, 1, 1)]:
            with self.subTest(coordinates=coordinates):
                with self.assertRaisesRegex(
                    ValueError,
                    exact(
                        f"Expected 'coordinates' to be a sequence of length "
                        f"{self.domain.dim} but received sequence of length "
                        f"{len(coordinates)}."
                    ),
                ):
                    self.domain.scale(coordinates)

    def test_scale_rescales_coordinates(self):
        """Test that each coordinate is rescaled linearly according to the bounds in
        the domain."""

        bounds = [(0, 1), (-0.5, 0.5), (1, 11)]
        domain = SimulatorDomain(bounds)
        coordinates = (0.5, 1, 0.7)
        transformed = domain.scale(coordinates)

        for z, x, bnds in zip(transformed, coordinates, bounds):
            with self.subTest(x=x, bnds=bnds, z=z):
                self.assertAlmostEqual(z, bnds[0] + x * (bnds[1] - bnds[0]))


if __name__ == "__main__":
    unittest.main()
