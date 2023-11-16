import itertools
import unittest

import numpy as np
from exauq.core.modelling import Input, Prediction, SimulatorDomain, TrainingDatum
from exauq.core.numerics import FLOAT_TOLERANCE, equal_within_tolerance
from tests.utilities.utilities import compare_input_tuples, exact, make_window


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
        """Test that the estimate and variance are of the same types as provided
        at initialisation."""

        estimates = [1, 1.2, np.float16(1.3)]
        variances = [1.2, np.float16(1.3), 1]

        for estimate, var in zip(estimates, variances):
            prediction = Prediction(estimate, var)
            self.assertIsInstance(prediction.estimate, type(estimate))
            self.assertIsInstance(prediction.variance, type(var))

    def test_non_real_error(self):
        """Test that a TypeError is raised if the supplied estimate or variance is not a
        real number."""

        for non_real in ["1", 1j]:
            with self.subTest(non_real=non_real):
                with self.assertRaisesRegex(
                    TypeError,
                    exact(
                        f"Expected 'estimate' to define a real number, but received {type(non_real)} "
                        "instead."
                    ),
                ):
                    Prediction(estimate=non_real, variance=1)

                with self.assertRaisesRegex(
                    TypeError,
                    exact(
                        f"Expected 'variance' to define a real number, but received {type(non_real)} "
                        "instead."
                    ),
                ):
                    Prediction(estimate=1, variance=non_real)

    def test_negative_variance_error(self):
        """Test that a ValueError is raised if a negative variance is provided at
        initialisation."""

        var = -0.1
        with self.assertRaisesRegex(
            ValueError,
            exact(f"'variance' must be a non-negative real number, but received {var}."),
        ):
            Prediction(estimate=1, variance=var)

    def test_immutable_fields(self):
        """Test that the estimate and variance values in a prediction are immutable."""

        prediction = Prediction(1, 1)
        with self.assertRaises(AttributeError):
            prediction.estimate = 0

        with self.assertRaises(AttributeError):
            prediction.variance = 0

    def test_equality_with_different_type(self):
        """Test that a Prediction object is not equal to an object of a different type."""

        p = Prediction(estimate=1, variance=1)
        for other in [1, "1", (1, 1)]:
            self.assertNotEqual(p, other)

    def test_equality_of_estimates(self):
        """Given two predictions with the same variances, test that they are equal if
        and only if the estimates are close according to the default tolerance."""

        variance = 0
        for estimate1 in [0.5 * n for n in range(-100, 101)]:
            p1 = Prediction(estimate1, variance)
            for estimate2 in make_window(estimate1, tol=FLOAT_TOLERANCE):
                p2 = Prediction(estimate2, variance)
                self.assertIs(p1 == p2, equal_within_tolerance(estimate1, estimate2))
                self.assertIs(p2 == p1, equal_within_tolerance(estimate1, estimate2))

    def test_equality_of_variances(self):
        """Given two predictions with the same estimates, test that they are equal if
        and only if the variances are close according to the default tolerance."""

        estimate = 0
        for var1 in [0.1 * n for n in range(101)]:
            p1 = Prediction(estimate, var1)
            for var2 in filter(lambda x: x >= 0, make_window(var1, tol=FLOAT_TOLERANCE)):
                p2 = Prediction(estimate, var2)
                self.assertIs(p1 == p2, equal_within_tolerance(var1, var2))
                self.assertIs(p2 == p1, p1 == p2)

    def test_equality_symmetric(self):
        """Test that equality of predictions doesn't depend on the order of the objects
        in the comparison."""

        for estimate1, var1 in itertools.product([-0.5, 0, 0.5], [0, 0.1, 0.2]):
            p1 = Prediction(estimate1, var1)
            estimates = make_window(estimate1, tol=FLOAT_TOLERANCE)
            variances = filter(lambda x: x >= 0, make_window(var1, tol=FLOAT_TOLERANCE))
            for estimate2, var2 in itertools.product(estimates, variances):
                p2 = Prediction(estimate2, var2)
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

    def test_init_with_valid_bounds(self):
        try:
            domain = SimulatorDomain([(0, 1), (-1, 1), (0, 100)])
        except Exception as e:
            self.fail(f"Initialisation with valid bounds failed with exception: {e}")

    def test_init_with_empty_bounds(self):
        with self.assertRaises(ValueError, msg="No ValueError raised for empty bounds"):
            SimulatorDomain([])

    def test_init_with_invalid_bounds_type(self):
        with self.assertRaises(TypeError, msg="No TypeError raised for invalid bounds type"):
            SimulatorDomain([(0, 1), "Invalid bounds", (0, 100)])

    def test_init_with_invalid_bound_length(self):
        with self.assertRaises(TypeError, msg="No TypeError raised for bound with invalid length"):
            SimulatorDomain([(0, 1, 2), (0, 1)])

    def test_init_with_non_real_numbers(self):
        with self.assertRaises(TypeError, msg="No TypeError raised for non-real numbers in bounds"):
            SimulatorDomain([(0, 1), (0, "1")])

    def test_init_with_low_greater_than_high(self):
        with self.assertRaises(ValueError, msg="No ValueError raised for low > high"):
            SimulatorDomain([(1, 0), (0, 1)])

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

    def test_get_corners_2d_domain(self):
        """Verify that get_corners accurately identifies and returns all four corners of a 2D
        unit square domain."""

        domain = SimulatorDomain([(0, 1), (0, 1)])
        corners = domain.get_corners()
        expected_corners = (Input(0, 0), Input(0, 1), Input(1, 0), Input(1, 1))
        self.assertEqual(corners, expected_corners)

    def test_get_corners_3d_domain(self):
        """This test validates that the get_corners method correctly generates and returns all
        the corner points of a three-dimensional domain, ensuring each corner is identified and
        returned properly."""

        domain = SimulatorDomain([(0, 1), (0, 1), (0, 1)])
        corners = domain.get_corners()
        expected_corners = (
            Input(0, 0, 0),
            Input(0, 0, 1),
            Input(0, 1, 0),
            Input(0, 1, 1),
            Input(1, 0, 0),
            Input(1, 0, 1),
            Input(1, 1, 0),
            Input(1, 1, 1),
        )
        self.assertEqual(corners, expected_corners)

    def test_get_corners_negative_bounds(self):
        """This test ensures that the `get_corners` method accurately identifies and returns all
        corner points for a two-dimensional domain with negative bounds, verifying its
        correctness when dealing with negative numbers."""

        domain = SimulatorDomain([(-1, 0), (-1, 0)])
        corners = domain.get_corners()
        expected_corners = (Input(-1, -1), Input(-1, 0), Input(0, -1), Input(0, 0))
        self.assertEqual(corners, expected_corners)

    def test_get_corners_2d_rectangle(self):
        """This test verifies that the `get_corners` method correctly calculates and returns all
        corner points for a two-dimensional rectangular domain, ensuring its functionality is not
        limited to square domains."""

        domain = SimulatorDomain([(0, 2), (0, 1)])
        corners = domain.get_corners()
        expected_corners = (Input(0, 0), Input(0, 1), Input(2, 0), Input(2, 1))
        self.assertEqual(corners, expected_corners)

    def test_get_corners_3d_rectangular_prism(self):
        """This test ensures the `get_corners` method accurately identifies all corners of a
        three-dimensional rectangular prism domain, showcasing its adaptability to handle domains
        of various shapes and dimensions."""

        domain = SimulatorDomain([(0, 2), (0, 1), (0, 3)])
        corners = domain.get_corners()
        expected_corners = (
            Input(0, 0, 0),
            Input(0, 0, 3),
            Input(0, 1, 0),
            Input(0, 1, 3),
            Input(2, 0, 0),
            Input(2, 0, 3),
            Input(2, 1, 0),
            Input(2, 1, 3),
        )
        self.assertEqual(corners, expected_corners)

    def test_get_corners_single_dimension(self):
        """This test verifies that the `get_corners` method correctly identifies the endpoints of
        a one-dimensional domain, demonstrating the method's capability to handle domains with a
        single dimension."""

        domain = SimulatorDomain([(0, 1)])
        corners = domain.get_corners()
        expected_corners = (Input(0), Input(1))
        self.assertEqual(corners, expected_corners)

    def test_get_corners_zero_width_bound(self):
        """This test ensures that the get_corners method accurately generates corner points for a
        domain with a zero-width bound in one dimension, demonstrating the method's robustness in
        handling edge cases."""

        domain = SimulatorDomain([(0, 0), (0, 1)])
        corners = domain.get_corners()
        expected_corners = (Input(0, 0), Input(0, 1))
        self.assertEqual(corners, expected_corners)

    def test_closest_boundary_points_basic(self):
        """This test checks the `closest_boundary_points` method for a straightforward scenario
        in a two-dimensional domain, ensuring that the method accurately finds the closest
        boundary points for an input point situated at the center of the domain."""

        domain = SimulatorDomain([(0, 1), (0, 1)])
        collection = [Input(0.5, 0.5)]
        result = domain.closest_boundary_points(collection)
        expected = (Input(0, 0.5), Input(1, 0.5), Input(0.5, 0), Input(0.5, 1))
        self.assertTupleEqual(
            result, expected, "Closest boundary points calculation is incorrect."
        )

    def test_closest_boundary_points_empty_collection(self):
        """This test validates that the `closest_boundary_points` method correctly handles an
        empty collection of points, issuing a user warning and returning an empty tuple as
        expected."""

        domain = SimulatorDomain([(0, 1), (0, 1)])
        collection = []
        with self.assertWarns(
            UserWarning, msg="No warning raised for empty collection."
        ):
            result = domain.closest_boundary_points(collection)
        self.assertEqual(
            result, tuple(), "Result should be an empty tuple for empty collection."
        )

    def test_closest_boundary_points_point_outside_domain(self):
        """This test ensures that the `closest_boundary_points` method appropriately handles
        situations where a point in the provided collection lies outside the domain. It checks
        that a user warning is issued to alert the user, and subsequently, a ValueError is raised
        to indicate the invalid input."""

        domain = SimulatorDomain([(0, 1), (0, 1)])
        collection = [Input(0.5, 0.5), Input(1.5, 0.5)]
        with self.assertWarns(
            UserWarning, msg="No warning raised for point outside domain."
        ):
            with self.assertRaises(
                ValueError, msg="No ValueError raised for point outside domain."
            ):
                domain.closest_boundary_points(collection)

    def test_closest_boundary_points_incorrect_dimensionality(self):
        """This test checks that the `closest_boundary_points` method correctly identifies and
        raises an error when a point in the collection does not match the domain's
        dimensionality. The domain is two-dimensional, but a three-dimensional point is provided
        in the collection, which should result in a ValueError being raised, ensuring the
        method's robustness to dimensionality mismatches."""

        domain = SimulatorDomain([(0, 1), (0, 1)])
        collection = [Input(0.5, 0.5, 0.5)]
        with self.assertRaises(
            ValueError,
            msg="No ValueError raised for point with incorrect dimensionality.",
        ):
            domain.closest_boundary_points(collection)

    def test_closest_boundary_points_on_boundary(self):
        """This test verifies that points on the boundary of a 2D domain are correctly identified
        as their own closest boundary points by the closest_boundary_points method."""

        domain = SimulatorDomain([(0, 1), (0, 1)])
        collection = [Input(0, 0.5), Input(1, 0.5), Input(0.5, 0), Input(0.5, 1)]
        result = domain.closest_boundary_points(collection)
        expected = tuple(collection)
        self.assertTupleEqual(
            result,
            expected,
            "Points on the boundary should be their own closest boundary points.",
        )

    def test_closest_boundary_points_high_dimensionality(self):
        """This test checks the `closest_boundary_points` method's ability to handle a
        high-dimensional space, ensuring it can accurately calculate the closest boundary points
        in a 10-dimensional domain."""

        bounds = [(0, 1)] * 10
        domain = SimulatorDomain(bounds)

        collection = [Input(*([0.5] * 10))]
        expected = []

        for i in range(10):
            modified_point_low = [0.5] * 10
            modified_point_low[i] = 0
            expected.append(Input(*modified_point_low))

            modified_point_high = [0.5] * 10
            modified_point_high[i] = 1
            expected.append(Input(*modified_point_high))

        result = domain.closest_boundary_points(collection)
        self.assertEqual(result, tuple(expected))

    def test_closest_boundary_points_float_precision(self):
        """This test verifies the `closest_boundary_points` method's precision and reliability
        when dealing with input points that have floating-point coordinates very close to
        boundary points, ensuring accurate calculations even in scenarios with potential
        floating-point precision challenges."""

        bounds = [(0, 1)] * 2
        domain = SimulatorDomain(bounds)

        collection = [Input(0.5000000001, 0.5000000001)]
        expected = (
            Input(0, 0.5000000001),
            Input(1, 0.5000000001),
            Input(0.5000000001, 0),
            Input(0.5000000001, 1),
        )

        result = domain.closest_boundary_points(collection)
        self.assertEqual(result, expected)

    def test_calculate_pseudopoints_basic(self):
        """This test checks if the calculate_pseudopoints method is working correctly by
        providing a basic 2D square domain and a collection of points inside it. It validates
        that the method calculates and returns the correct boundary and corner pseudopoints."""

        domain = SimulatorDomain([(0, 1), (0, 1)])
        collection = [Input(0.25, 0.25), Input(0.75, 0.75)]
        pseudopoints = domain.calculate_pseudopoints(collection)
        expected = (
            Input(0, 0.25),
            Input(1, 0.75),
            Input(0.25, 0),
            Input(0.75, 1),
            Input(0, 0),
            Input(0, 1),
            Input(1, 0),
            Input(1, 1),
        )
        self.assertEqual(
            pseudopoints, expected, "Pseudopoints calculation is incorrect."
        )

    def test_calculate_pseudopoints_empty_collection(self):
        """This test ensures that if an empty collection is provided to the
        calculate_pseudopoints method, it correctly warns the user and returns only the corner
        pseudopoints."""

        domain = SimulatorDomain([(0, 1), (0, 1)])
        collection = []
        with self.assertWarns(
            UserWarning, msg="No warning raised for empty collection."
        ):
            pseudopoints = domain.calculate_pseudopoints(collection)
        expected = (Input(0, 0), Input(0, 1), Input(1, 0), Input(1, 1))
        self.assertEqual(
            pseudopoints,
            expected,
            "Pseudopoints should only include corner points for empty collection.",
        )

    def test_calculate_pseudopoints_point_outside_domain(self):
        """This test verifies that the calculate_pseudopoints method properly handles points
        outside the domain by raising a ValueError and issuing a warning."""

        domain = SimulatorDomain([(0, 1), (0, 1)])
        collection = [Input(0.5, 0.5), Input(1.5, 0.5)]
        with self.assertWarns(
            UserWarning, msg="No warning raised for point outside domain."
        ):
            with self.assertRaises(
                ValueError, msg="No ValueError raised for point outside domain."
            ):
                domain.calculate_pseudopoints(collection)

    def test_calculate_pseudopoints_incorrect_dimensionality(self):
        """This test ensures that if a point with incorrect dimensionality is passed to the
        calculate_pseudopoints method, a ValueError is raised."""

        domain = SimulatorDomain([(0, 1), (0, 1)])
        collection = [Input(0.5, 0.5, 0.5)]
        with self.assertRaises(
            ValueError,
            msg="No ValueError raised for point with incorrect dimensionality.",
        ):
            domain.calculate_pseudopoints(collection)

    def test_calculate_pseudopoints_high_dimensionality(self):
        """This test checks the calculate_pseudopoints method's performance and accuracy in a
        high-dimensional space, ensuring it correctly calculates boundary and corner
        pseudopoints."""

        bounds = [(0, 1)] * 10
        domain = SimulatorDomain(bounds)
        collection = [Input(*([0.5] * 10))]
        pseudopoints = domain.calculate_pseudopoints(collection)
        expected_boundary_points = []
        for i in range(10):
            low = [0.5] * 10
            low[i] = 0
            expected_boundary_points.append(Input(*low))
            high = [0.5] * 10
            high[i] = 1
            expected_boundary_points.append(Input(*high))
        expected_corners = domain.get_corners()
        expected = tuple(expected_boundary_points + list(expected_corners))
        self.assertEqual(
            pseudopoints,
            expected,
            "Pseudopoints calculation is incorrect for high-dimensional space.",
        )


if __name__ == "__main__":
    unittest.main()
