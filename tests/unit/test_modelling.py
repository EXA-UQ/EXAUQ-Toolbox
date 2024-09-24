import copy
import itertools
import math
import pathlib
import tempfile
import unittest
from numbers import Real
from typing import Literal, Sequence

import numpy as np

from exauq.core.emulators import MogpEmulator, MogpHyperparameters
from exauq.core.modelling import (
    AbstractGaussianProcess,
    GaussianProcessHyperparameters,
    GaussianProcessPrediction,
    Input,
    LevelTagged,
    MultiLevel,
    MultiLevelGaussianProcess,
    OptionalFloatPairs,
    Prediction,
    SimulatorDomain,
    TrainingDatum,
    _LevelTaggedOld,
    get_level,
    remove_level,
    set_level,
)
from exauq.core.numerics import FLOAT_TOLERANCE, equal_within_tolerance
from exauq.utilities.csv_db import Path
from tests.unit.fakes import FakeGP, WhiteNoiseGP, WhiteNoiseGPHyperparameters
from tests.utilities.utilities import (
    ExauqTestCase,
    compare_input_tuples,
    exact,
    make_window,
)


class TestInput(unittest.TestCase):
    def test_input_reals(self):
        """Test that an input can be constructed from args that are
        real numbers."""
        _ = Input(np.float16(1.2345))
        _ = Input(1.2, np.int32(33))

    def test_input_non_real_error(self):
        """Test that TypeError is raised during construction if there is an
        arg that doesn't define a real number."""

        msg = "Arguments must be instances of real numbers"
        for coord in ["a", complex(1, 1)]:
            with self.subTest(coord=coord):
                with self.assertRaisesRegex(TypeError, exact(msg)):
                    Input(1, coord)

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
            _ = Input(1.1, -np.inf)

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
            _ = Input.from_array(np.array([1.1, -np.inf]))

        self.assertEqual(msg, str(cm.exception))


class TestTrainingDatum(unittest.TestCase):
    def setUp(self) -> None:
        # Temp directory for holding csv file during a unit test run
        self._dir = tempfile.TemporaryDirectory()
        self.tmp_dir = self._dir.name
        self.path = pathlib.Path(self.tmp_dir, "data.csv")

    def tearDown(self) -> None:
        self._dir.cleanup()

    @staticmethod
    def write_csv_data(path: Path, data: Sequence[Sequence[str]], mode="x"):
        """Write data to a csv file."""

        with open(path, mode=mode, newline="") as f:
            for row in data:
                f.write(",".join(map(str, row)) + "\n")

    def test_input_error(self):
        """Test that a TypeError is raised if the constructor arg 'input'
        is not an Input."""

        msg = "Argument 'input' must be of type Input"
        with self.assertRaisesRegex(TypeError, exact(msg)):
            TrainingDatum(1, 1)

    def test_output_error(self):
        """Test that a TypeError is raised if the constructor arg 'output'
        is not a real number."""

        msg = "Argument 'output' must define a real number"
        for output in ["a", complex(1, 1), [1.1]]:
            with self.subTest(output=output):
                with self.assertRaisesRegex(TypeError, exact(msg)):
                    TrainingDatum(Input(1), output)

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
            _ = TrainingDatum(Input(1), -np.inf)

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
        with self.assertRaisesRegex(AttributeError, "cannot assign to field 'input'$"):
            datum.input = Input(2)

        with self.assertRaisesRegex(AttributeError, "cannot assign to field 'output'$"):
            datum.output = 1

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

    def test_read_from_csv_default_output_column(self):
        """By default, the csv data is read into a sequence of training data where the
        last column in the csv file defines the simulator outputs and the other columns
        define simulator inputs."""

        self.write_csv_data(self.path, [[1, 2, 3], [10, 20, 30]])
        training_data = TrainingDatum.read_from_csv(self.path)
        expected = (TrainingDatum(Input(1, 2), 3), TrainingDatum(Input(10, 20), 30))
        self.assertEqual(expected, training_data)

    def test_read_from_csv_output_column_specified(self):
        """The column of the simulator outputs can be specified as a zero-based index.
        The remaining columns are then interpreted as simulator inputs (in the order they
        appear in the file)."""

        self.write_csv_data(self.path, [[1, 2, 3], [10, 20, 30]])
        training_data = TrainingDatum.read_from_csv(self.path, output_col=1)
        expected = (TrainingDatum(Input(1, 3), 2), TrainingDatum(Input(10, 30), 20))
        self.assertEqual(expected, training_data)

    def test_read_from_csv_header_row_skipped(self):
        """If the csv file contains a header row, then this is skipped when reading the
        data."""

        self.write_csv_data(self.path, [["x1", "x2", "y"], [1, 2, 3], [10, 20, 30]])
        training_data = TrainingDatum.read_from_csv(self.path, header=True)
        expected = (TrainingDatum(Input(1, 2), 3), TrainingDatum(Input(10, 20), 30))
        self.assertEqual(expected, training_data)

    def test_read_from_csv_empty_data(self):
        """If the csv file does not contain any training data, then an empty tuple is
        returned."""

        # No header row case
        self.write_csv_data(self.path, [[]])
        training_data = TrainingDatum.read_from_csv(self.path)
        self.assertEqual(tuple(), training_data)

        # Header row case
        self.write_csv_data(self.path, [["x", "y"], []], mode="w")
        training_data = TrainingDatum.read_from_csv(self.path, header=True)
        self.assertEqual(tuple(), training_data)

        # Header row case with empty file
        self.write_csv_data(self.path, [], mode="w")
        training_data = TrainingDatum.read_from_csv(self.path, header=True)
        self.assertEqual(tuple(), training_data)

    def test_read_from_csv_cannot_parse_float_error(self):
        """An AssertionError is raised if any of the read data cannot be parsed as a
        float."""

        # Bad simulator input
        bad_data = "foo"
        self.write_csv_data(self.path, [[1, 2, 3], [10, bad_data, 30]])
        with self.assertRaisesRegex(
            AssertionError,
            exact(
                f"Could not read data from {self.path}: unable to parse value '{bad_data}' as a float."
            ),
        ):
            _ = TrainingDatum.read_from_csv(self.path)

        # Bad simulator output
        bad_data = ""
        self.write_csv_data(self.path, [[1, 2, bad_data], [10, 20, 30]], mode="w")
        with self.assertRaisesRegex(
            AssertionError,
            exact(
                f"Could not read data from {self.path}: unable to parse value '{bad_data}' as a float."
            ),
        ):
            _ = TrainingDatum.read_from_csv(self.path)

    def test_read_from_csv_cannot_parse_inf_and_NaN_error(self):
        """An AssertionError is raised if any of the data contains infinite or NaN
        values."""

        # Infinite value
        self.write_csv_data(self.path, [[1, 2, 3], [10, "inf", 30]])
        with self.assertRaisesRegex(
            AssertionError,
            exact(f"Could not read data from {self.path}: infinite or NaN values found."),
        ):
            _ = TrainingDatum.read_from_csv(self.path)

        # NaN value
        self.write_csv_data(self.path, [[1, 2, 3], [10, "NaN", 30]], mode="w")
        with self.assertRaisesRegex(
            AssertionError,
            exact(f"Could not read data from {self.path}: infinite or NaN values found."),
        ):
            _ = TrainingDatum.read_from_csv(self.path)

    def test_read_from_csv_bad_column_index_error(self):
        """A ValueError is raised if an out-of-bounds column index is given, where the
        index bounds are determined row-wise."""

        # Square data case
        data = [[1, 2, 3], [10, 20, 30]]
        self.write_csv_data(self.path, data)
        bad_row = 0
        for output_col in [-4, 3]:
            with self.assertRaisesRegex(
                ValueError,
                exact(
                    f"'output_col={output_col}' does not define a valid column index for "
                    f"csv data with {len(data[bad_row])} columns in row {bad_row}."
                ),
            ):
                _ = TrainingDatum.read_from_csv(self.path, output_col=output_col)

        # Jagged shape case - row 1 the culprit
        data = [[1, 2, 3], [10, 30]]
        self.write_csv_data(self.path, data, mode="w")
        output_col = 2
        bad_row = 1
        with self.assertRaisesRegex(
            ValueError,
            exact(
                f"'output_col={output_col}' does not define a valid column index for "
                f"csv data with {len(data[bad_row])} columns in row {bad_row}."
            ),
        ):
            _ = TrainingDatum.read_from_csv(self.path, output_col=output_col)


class TestPrediction(ExauqTestCase):
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
        """Test that the estimate, variance and standard_deviation values in a prediction are immutable."""

        prediction = Prediction(1, 1)
        with self.assertRaises(AttributeError):
            prediction.estimate = 0

        with self.assertRaises(AttributeError):
            prediction.variance = 0

        with self.assertRaises(AttributeError):
            prediction.standard_deviation = 0

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

    def test_standard_deviation_calculation(self):
        """Test that the standard_deviation is correctly calculated as the square root of the variance."""
        for variance in [0, 0.25, 1, 2, 4, 9]:
            with self.subTest(variance=variance):
                prediction = Prediction(estimate=1, variance=variance)
                expected_std = math.sqrt(variance)
                self.assertEqualWithinTolerance(
                    prediction.standard_deviation, expected_std
                )

    def test_standard_deviation_invariance_to_estimate(self):
        """Test that the standard_deviation does not change with different estimates."""
        variance = 2.5
        std = math.sqrt(variance)
        for estimate in range(-10, 11):
            prediction = Prediction(estimate=estimate, variance=variance)
            self.assertEqual(prediction.standard_deviation, std)

    def test_standard_deviation_zero_variance(self):
        """Test that the standard_deviation is zero when variance is zero."""
        prediction = Prediction(estimate=5, variance=0)
        self.assertEqual(prediction.standard_deviation, 0)


class TestGaussianProcessPrediction(ExauqTestCase):
    def test_nes_error_arg_type_error(self):
        """A TypeError is raised when computing the normalised expected square error if
        the observed output is not a Real number.
        """

        observed_output = "1"

        prediction = GaussianProcessPrediction(estimate=1, variance=1)

        with self.assertRaisesRegex(
            TypeError,
            exact(
                f"Expected 'observed_output' to be of type {Real} but received type {type(observed_output)}."
            ),
        ):
            prediction.nes_error(observed_output)

    def test_nes_error_value_error_raised_if_observed_output_is_infinite(self):
        """A ValueError is raised if the observed output is an infinite value or NaN."""

        prediction = GaussianProcessPrediction(estimate=1, variance=1)

        for observed_output in [np.nan, np.inf, -np.inf]:
            with self.subTest(observed_output=observed_output), self.assertRaisesRegex(
                ValueError,
                exact(
                    f"'observed_output' must be a finite real number, but received {observed_output}."
                ),
            ):
                prediction.nes_error(observed_output)

    def test_nes_error_formula(self):
        """The normalised expected square error is given by the expected square error
        divided by the standard deviation of the square error, as described in
        Mohammadi et al (2022).
        """

        variances = [0.1, 0.2, 0.3]
        means = [-1, 0, 1]
        observed_outputs = [0.9, -0.1, 0, 10]
        for mean, var, observed_output in itertools.product(
            means, variances, observed_outputs
        ):
            with self.subTest(mean=mean, var=var, observed_output=observed_output):
                prediction = GaussianProcessPrediction(estimate=1, variance=var)
                square_err = (prediction.estimate - observed_output) ** 2
                expected_sq_err = prediction.variance + square_err
                standard_deviation_sq_err = math.sqrt(
                    2 * (prediction.variance**2) + 4 * prediction.variance * square_err
                )

                self.assertEqualWithinTolerance(
                    expected_sq_err / standard_deviation_sq_err,
                    prediction.nes_error(observed_output),
                )

    def test_nes_error_zero_variance_cases(self):
        """The normalised expected square error is equal to

        * zero if the variance of the prediction is zero and the observed output is equal
          to the prediction's estimate.
        * inf if the variance of the prediction is zero and the observed output is not
          equal to the prediction's estimate.
        """

        mean = 1

        self.assertEqual(
            0, GaussianProcessPrediction(estimate=mean, variance=0).nes_error(mean)
        )

        self.assertEqual(
            float("inf"),
            GaussianProcessPrediction(estimate=mean, variance=0).nes_error(mean + 1e-5),
        )


class TestAbstractGaussianProcess(ExauqTestCase):
    def setUp(self) -> None:
        self.emulator = FakeGP()
        self.training_data = [TrainingDatum(Input(0.5), 1)]
        self.emulator.fit(self.training_data)

        self.inputs = [Input(0), Input(0.25), Input(1)]
        self.outputs = [-1, np.int32(1), 2.1, np.float16(3)]

    def test_covariance_matrix_correlations_with_training_data(self):
        """The covariance matrix consists of the correlations of the supplied inputs
        with the training data inputs for the GP."""

        emulator = MogpEmulator()
        training_inputs = [Input(0.2, 0.3), Input(0.4, 0.1), Input(0.6, 0.8)]
        params = MogpHyperparameters(
            corr_length_scales=[0.5, 0.5], process_var=1, nugget=None
        )
        emulator.fit(
            [TrainingDatum(x, 1) for x in training_inputs],
            hyperparameters=params,
        )
        inputs = [Input(0.2, 1), Input(0.8, -0.5)]

        self.assertArraysEqual(
            emulator.correlation(inputs, training_inputs),
            emulator.covariance_matrix(inputs),
        )


class TestGaussianProcessHyperparameters(ExauqTestCase):
    def setUp(self) -> None:
        # N.B. although single-element Numpy arrays can be converted to scalars this is
        # deprecated functionality and will throw an error in the future.
        self.nonreal_objects = [2j, "1", np.array([2])]
        self.negative_reals = [-0.5, -math.inf]
        self.nonpositive_reals = self.negative_reals + [0]
        self.hyperparameters = {
            "correlation": {
                "func": GaussianProcessHyperparameters.transform_corr,
                "arg": "corr_length_scales",
            },
            "variance": {
                "func": GaussianProcessHyperparameters.transform_cov,
                "arg": "process_var",
            },
            "nugget": {
                "func": GaussianProcessHyperparameters.transform_nugget,
                "arg": "nugget",
            },
        }

    def test_init_checks_arg_types(self):
        """A TypeError is raised upon initialisation if:

        * the correlations is not a sequence or Numpy array; or
        * the process variance is not a real number; or
        * the nugget is not ``None`` or a real number.
        """

        # correlations
        nonseq_objects = [1.0, {1.0}]
        for corr in nonseq_objects:
            with self.subTest(corr=corr), self.assertRaisesRegex(
                TypeError,
                exact(
                    f"Expected 'corr_length_scales' to be a sequence or Numpy array, but received {type(corr)}."
                ),
            ):
                _ = GaussianProcessHyperparameters(
                    corr_length_scales=corr, process_var=1.0, nugget=1.0
                )

        # process variance
        for cov in self.nonreal_objects + [None]:
            with self.subTest(cov=cov), self.assertRaisesRegex(
                TypeError,
                exact(
                    f"Expected 'process_var' to be a real number, but received {type(cov)}."
                ),
            ):
                _ = GaussianProcessHyperparameters(
                    corr_length_scales=[1.0], process_var=cov, nugget=1.0
                )

        # nugget
        for nugget in self.nonreal_objects:
            with self.subTest(nugget=nugget), self.assertRaisesRegex(
                TypeError,
                exact(
                    f"Expected 'nugget' to be a real number, but received {type(nugget)}."
                ),
            ):
                _ = GaussianProcessHyperparameters(
                    corr_length_scales=[1.0], process_var=1.0, nugget=nugget
                )

    def test_init_checks_arg_values(self):
        """A ValueError is raised upon initialisation if:

        * the correlation is not a sequence / array of positive real numbers; or
        * the process variance is not a positive real number; or
        * the nugget is < 0 (if not None).
        """

        # correlations
        bad_values = [[x] for x in self.nonreal_objects + self.nonpositive_reals]
        for corr in bad_values:
            with self.subTest(corr=corr), self.assertRaisesRegex(
                ValueError,
                exact(
                    "Expected 'corr_length_scales' to be a sequence or Numpy array of positive real numbers, "
                    f"but found element {corr[0]} of type {type(corr[0])}."
                ),
            ):
                _ = GaussianProcessHyperparameters(
                    corr_length_scales=corr, process_var=1.0, nugget=1.0
                )

        # process variance
        for cov in self.nonpositive_reals:
            with self.subTest(cov=cov), self.assertRaisesRegex(
                ValueError,
                exact(
                    f"Expected 'process_var' to be a positive real number, but received {cov}."
                ),
            ):
                _ = GaussianProcessHyperparameters(
                    corr_length_scales=[1.0], process_var=cov, nugget=1.0
                )

        # nugget
        for nugget in self.negative_reals:
            with self.subTest(nugget=nugget), self.assertRaisesRegex(
                ValueError,
                exact(
                    f"Expected 'nugget' to be a positive real number, but received {nugget}."
                ),
            ):
                _ = GaussianProcessHyperparameters(
                    corr_length_scales=[1.0], process_var=1.0, nugget=nugget
                )

    def test_transformation_formulae(self):
        """The transformed correlation is equal to `-2 * log(corr)`.
        The transformed process variance is equal to `log(cov)`.
        The transformed nugget is equal to `log(nugget)`."""

        positive_reals = [0.1, 1, 10, np.float16(1.1)]
        for x in positive_reals:
            with self.subTest(hyperparameter="correlation", x=x):
                transformation_func = self.hyperparameters["correlation"]["func"]
                self.assertEqual(-2 * math.log(x), transformation_func(x))

            with self.subTest(hyperparameter="variance", x=x):
                transformation_func = self.hyperparameters["variance"]["func"]
                self.assertEqual(math.log(x), transformation_func(x))

            with self.subTest(hyperparameter="nugget", x=x):
                transformation_func = self.hyperparameters["nugget"]["func"]
                self.assertEqual(math.log(x), transformation_func(x))

    def test_equals_hyperparameter_values(self):
        """Two instances of GaussianProcessHyperparameters are considered equal precisely
        when all of the following hold:

        * The correlation length scales are equal within the default tolerance.
        * The process variances are equal within the default tolerance.
        * The nuggets are real numbers equal within the default tolerances, or are both
          None.
        """

        epsilon = 0.75 * FLOAT_TOLERANCE

        args1 = {
            "corr_length_scales": np.array([1.1, 1.2]),
            "process_var": 1,
            "nugget": 0.5,
        }
        hyperparameters1 = GaussianProcessHyperparameters(**args1)

        for arg in ["corr_length_scales", "process_var", "nugget"]:
            # Equality within tolerances
            args2 = copy.deepcopy(args1)
            args2[arg] = args1[arg] + epsilon
            self.assertEqual(
                hyperparameters1,
                GaussianProcessHyperparameters(**args2),
            )

            # Not equal outside of tolerances
            args2[arg] += epsilon
            self.assertNotEqual(
                hyperparameters1,
                GaussianProcessHyperparameters(**args2),
            )

        # nugget = None cases
        self.assertEqual(
            GaussianProcessHyperparameters(
                corr_length_scales=[1], process_var=2, nugget=None
            ),
            GaussianProcessHyperparameters(
                corr_length_scales=[1], process_var=2, nugget=None
            ),
        )
        self.assertNotEqual(
            GaussianProcessHyperparameters(
                corr_length_scales=[1], process_var=2, nugget=3
            ),
            GaussianProcessHyperparameters(
                corr_length_scales=[1], process_var=2, nugget=None
            ),
        )
        self.assertNotEqual(
            GaussianProcessHyperparameters(
                corr_length_scales=[1], process_var=2, nugget=None
            ),
            GaussianProcessHyperparameters(
                corr_length_scales=[1], process_var=2, nugget=3
            ),
        )

    def test_equals_not_true_if_different_types(self):
        """A GaussianProcessHyperparameters instance is not equal to an object of a
        different type."""

        hyperparameters = GaussianProcessHyperparameters([1], 1, 1)
        self.assertNotEqual(hyperparameters, "a")
        self.assertNotEqual("a", hyperparameters)

    def test_transformations_of_limit_values(self):
        """The transformation functions handle limit values of their domains in the
        following ways:

        * For correlations, `inf` maps to `-inf` and `0` maps to `inf`.
        * For process variances and nuggets, `inf` maps to `inf` and 0 maps to `-inf`.
        """

        with self.subTest(hyperparameter="correlation"):
            transformation_func = self.hyperparameters["correlation"]["func"]
            self.assertEqual(-math.inf, transformation_func(math.inf))
            self.assertEqual(math.inf, transformation_func(0))

        with self.subTest(hyperparameter="variance"):
            transformation_func = self.hyperparameters["variance"]["func"]
            self.assertEqual(math.inf, transformation_func(math.inf))
            self.assertEqual(-math.inf, transformation_func(0))

        with self.subTest(hyperparameter="nugget"):
            transformation_func = self.hyperparameters["nugget"]["func"]
            self.assertEqual(math.inf, transformation_func(math.inf))
            self.assertEqual(-math.inf, transformation_func(0))

    def test_transforms_non_real_arg_raises_type_error(self):
        """A TypeError is raised if the argument supplied is not a real number."""

        for hyperparameter, x in itertools.product(
            self.hyperparameters, self.nonreal_objects
        ):
            arg = self.hyperparameters[hyperparameter]["arg"]
            transformation_func = self.hyperparameters[hyperparameter]["func"]
            with self.subTest(hyperparameter=hyperparameter, x=x), self.assertRaisesRegex(
                TypeError,
                exact(f"Expected '{arg}' to be a real number, but received {type(x)}."),
            ):
                _ = transformation_func(x)

    def test_transforms_with_nonpositive_value_raises_value_error(self):
        """A ValueError is raised if the argument supplied is < 0."""

        for hyperparameter, x in itertools.product(
            self.hyperparameters, self.negative_reals
        ):
            arg = self.hyperparameters[hyperparameter]["arg"]
            transformation_func = self.hyperparameters[hyperparameter]["func"]
            with self.subTest(hyperparameter=hyperparameter, x=x), self.assertRaisesRegex(
                ValueError,
                exact(f"'{arg}' cannot be < 0, but received {x}."),
            ):
                _ = transformation_func(x)


class TestSimulatorDomain(unittest.TestCase):
    def setUp(self) -> None:
        self.epsilon = FLOAT_TOLERANCE / 2  # Useful for testing equality up to tolerance
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
            _ = SimulatorDomain([(0, 1), (-1, 1), (0, 100)])
        except Exception as e:
            self.fail(f"Initialisation with valid bounds failed with exception: {e}")

    def test_init_with_empty_bounds(self):
        with self.assertRaises(
            ValueError, msg="No ValueError raised for empty bounds"
        ) as context:
            SimulatorDomain([])

        self.assertEqual(
            str(context.exception), "At least one pair of bounds must be provided."
        )

    def test_init_with_none_bounds(self):
        with self.assertRaises(
            TypeError, msg="No TypeError raised for None instead of valid bounds"
        ) as context:
            SimulatorDomain(None)

        self.assertEqual(
            str(context.exception),
            "Bounds cannot be None. 'bounds' should be a sequence.",
        )

    def test_init_with_non_ordered_bounds(self):
        with self.assertRaises(
            TypeError, msg="No TypeError raised for non-ordered collection."
        ) as context:
            SimulatorDomain({(0, 1), (0, 1)})

        self.assertEqual(str(context.exception), "Bounds should be a sequence.")

    def test_init_with_invalid_bounds_type(self):
        with self.assertRaises(
            ValueError, msg="No ValueError raised for invalid bounds type"
        ) as context:
            SimulatorDomain([(0, 1), "Invalid bounds", (0, 100)])

        self.assertEqual(
            str(context.exception), "Each bound must be a tuple of two numbers."
        )

    def test_init_with_invalid_bound_length(self):
        test_cases = [
            ([(0, 1, 2), (0, 1)], "Case with a 3-tuple and a 2-tuple"),
            ([(0, 1), (0, 1, 2)], "Case with a 2-tuple and a 3-tuple"),
        ]

        for bounds, msg in test_cases:
            with self.subTest(msg=msg):
                with self.assertRaises(
                    ValueError, msg=f"No ValueError raised for bounds: {bounds}"
                ) as context:
                    SimulatorDomain(bounds)

                self.assertEqual(
                    str(context.exception), "Each bound must be a tuple of two numbers."
                )

    def test_init_with_non_real_numbers(self):
        with self.assertRaises(
            TypeError, msg="No TypeError raised for non-real numbers in bounds"
        ) as context:
            SimulatorDomain([(0, 1), (0, "1")])

        self.assertEqual(str(context.exception), "Bounds must be real numbers.")

    def test_init_with_low_greater_than_high(self):
        test_cases = [
            ([(1, 0), (0, 1)], "Case with positive (<high>, <low>), (<low>, <high>)"),
            ([(0, 1), (1, 0)], "Case with positive (<low>, <high>), (<high>, <low>)"),
        ]

        for bounds, msg in test_cases:
            with self.subTest(msg=msg):
                with self.assertRaises(
                    ValueError,
                    msg=f"No ValueError raised for low > high bounds: {bounds}",
                ) as context:
                    SimulatorDomain(bounds)

                self.assertEqual(
                    str(context.exception),
                    "Lower bound cannot be greater than upper bound.",
                )

    def test_init_with_low_greater_than_high_but_within_tolerance(self):
        """Test that a SimulatorDomain can be initialised with bounds even if one set of
        bounds has the lower bound greater than the upper bound but within the tolerance
        for equality."""

        try:
            _ = SimulatorDomain([(0 + self.epsilon, 0)])
        except Exception:
            raise self.fail("Should not have raised an exception")

    def test_dim_equal_number_of_supplied_bounds(self):
        """Test that the dimension of the domain is equal to the length of the bounds
        sequence that was supplied."""

        self.assertEqual(len(self.bounds), self.domain.dim)

    def test_dim_immutable(self):
        """Test that the dim property is read-only."""

        with self.assertRaises(AttributeError):
            self.domain.dim = 1

    def test_bounds_returns_supplied_bounds(self):
        """Test that the bounds supplied at initialisation are returned as a
        tuple of tuples."""

        self.assertEqual(tuple(self.bounds), self.domain.bounds)

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
        corners = domain.get_corners
        expected_corners = (Input(0, 0), Input(0, 1), Input(1, 0), Input(1, 1))
        self.assertTrue(compare_input_tuples(corners, expected_corners))

    def test_get_corners_3d_domain(self):
        """This test validates that the get_corners method correctly generates and returns all
        the corner points of a three-dimensional domain, ensuring each corner is identified and
        returned properly."""

        domain = SimulatorDomain([(0, 1), (0, 1), (0, 1)])
        corners = domain.get_corners
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
        self.assertTrue(compare_input_tuples(corners, expected_corners))

    def test_get_corners_negative_bounds(self):
        """This test ensures that the `get_corners` method accurately identifies and returns all
        corner points for a two-dimensional domain with negative bounds, verifying its
        correctness when dealing with negative numbers."""

        domain = SimulatorDomain([(-1, 0), (-1, 0)])
        corners = domain.get_corners
        expected_corners = (Input(-1, -1), Input(-1, 0), Input(0, -1), Input(0, 0))
        self.assertTrue(compare_input_tuples(corners, expected_corners))

    def test_get_corners_2d_rectangle(self):
        """This test verifies that the `get_corners` method correctly calculates and returns all
        corner points for a two-dimensional rectangular domain, ensuring its functionality is not
        limited to square domains."""

        domain = SimulatorDomain([(0, 2), (0, 1)])
        corners = domain.get_corners
        expected_corners = (Input(0, 0), Input(0, 1), Input(2, 0), Input(2, 1))
        self.assertTrue(compare_input_tuples(corners, expected_corners))

    def test_get_corners_3d_rectangular_prism(self):
        """This test ensures the `get_corners` method accurately identifies all corners of a
        three-dimensional rectangular prism domain, showcasing its adaptability to handle domains
        of various shapes and dimensions."""

        domain = SimulatorDomain([(0, 2), (0, 1), (0, 3)])
        corners = domain.get_corners
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
        self.assertTrue(compare_input_tuples(corners, expected_corners))

    def test_get_corners_single_dimension(self):
        """This test verifies that the `get_corners` method correctly identifies the endpoints of
        a one-dimensional domain, demonstrating the method's capability to handle domains with a
        single dimension."""

        domain = SimulatorDomain([(0, 1)])
        corners = domain.get_corners
        expected_corners = (Input(0), Input(1))
        self.assertTrue(compare_input_tuples(corners, expected_corners))

    def test_get_corners_zero_width_bound(self):
        """This test ensures that the get_corners method accurately generates corner points for a
        domain with a zero-width bound in one dimension, demonstrating the method's robustness in
        handling edge cases."""

        domain = SimulatorDomain([(0, 0), (0, 1)])
        corners = domain.get_corners
        expected_corners = (Input(0, 0), Input(0, 1))
        self.assertTrue(compare_input_tuples(corners, expected_corners))

    def test_get_corners_identifies_corners_that_are_equal_within_tolerance(self):
        """Test that if the bounds on a dimension are within the standard tolerance of
        each other, then there is only one corner coordinate for that dimension."""

        domain = SimulatorDomain([(0, 1), (0, self.epsilon)])
        corners = domain.get_corners
        self.assertEqual(2, len(corners))
        for corner in corners:
            self.assertTrue(equal_within_tolerance(0, corner[1]))

    def test_closest_boundary_points_basic(self):
        """This test checks the `closest_boundary_points` method for a straightforward scenario
        in a two-dimensional domain, ensuring that the method accurately finds the closest
        boundary points for an input point situated at the center of the domain."""

        domain = SimulatorDomain([(0, 1), (0, 1)])
        collection = [Input(0.5, 0.5)]
        result = domain.closest_boundary_points(collection)
        expected = (Input(0, 0.5), Input(1, 0.5), Input(0.5, 0), Input(0.5, 1))
        self.assertTrue(
            compare_input_tuples(result, expected),
            "Closest boundary points calculation is incorrect.",
        )

    def test_closest_boundary_points_empty_collection(self):
        """This test validates that the `closest_boundary_points` method correctly handles an
        empty collection of points and returns an empty tuple as expected."""

        domain = SimulatorDomain([(0, 1), (0, 1)])
        collection = []
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

        self.assertTrue(
            compare_input_tuples(result, expected),
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
        self.assertTrue(compare_input_tuples(result, tuple(expected)))

    def test_closest_boundary_points_float_precision(self):
        """This test verifies the `closest_boundary_points` method's precision and reliability
        when dealing with input points that have floating-point coordinates very close to
        boundary points, ensuring accurate calculations even in scenarios with potential
        floating-point precision challenges."""

        x = (1 + 10 * FLOAT_TOLERANCE) * 0.5
        bounds = [(0, 1)] * 2
        domain = SimulatorDomain(bounds)

        collection = [Input(x, x)]
        expected = (
            Input(0, x),
            Input(1, x),
            Input(x, 0),
            Input(x, 1),
        )

        result = domain.closest_boundary_points(collection)
        self.assertTrue(compare_input_tuples(result, expected))

    def test_closest_boundary_points_interior_points(self):
        """This test checks if the `closest_boundary_points` method handles 'interior' points
        correctly."""

        bounds = [(0, 1)] * 2
        domain = SimulatorDomain(bounds)

        collection = [Input(0.4, 0.8), Input(0.5, 0.5), Input(0.7, 0.2)]
        expected = (
            Input(0, 0.8),
            Input(1, 0.2),
            Input(0.7, 0),
            Input(0.4, 1),
        )

        result = domain.closest_boundary_points(collection)
        self.assertTrue(compare_input_tuples(result, expected))

    def test_closest_boundary_points_identifies_points_that_are_equal_within_tolerance(
        self,
    ):
        """Test that, if two boundary points are equal within the standard tolerance,
        then they get identified in the output."""

        domain = SimulatorDomain([(0, 1), (0, self.epsilon)])
        inputs = [Input(0.5, 0), Input(0.5, self.epsilon)]
        boundary_points = domain.closest_boundary_points(inputs)
        self.assertEqual(1, len(boundary_points))
        self.assertTrue(
            equal_within_tolerance(Input(0.5, self.epsilon), boundary_points[0])
        )

    def test_closest_boundary_points_returns_one_point_per_boundary(self):
        """Test that there is one exactly one point returned per boundary in the case
        where all points are closer to one particular coordinate slice than the other."""

        bounds = [(0, 1), (-1, 1)]
        domain = SimulatorDomain(bounds)
        inputs = [Input(0.9, 0.2), Input(0.9, 0.8)]
        boundary_points = domain.closest_boundary_points(inputs)

        self.assertEqual(4, len(boundary_points))

        def is_on_boundary(
            x: Input, coordinate: int, limit: Literal["lower", "upper"]
        ) -> bool:
            if limit == "lower":
                return x[coordinate] == bounds[coordinate][0]
            elif limit == "upper":
                return x[coordinate] == bounds[coordinate][1]
            else:
                raise ValueError("'bound' should be one of 'lower' and 'upper'")

        for coord, limit in itertools.product([0, 1], ["upper", "lower"]):
            with self.subTest(coord=coord, limit=limit):
                self.assertEqual(
                    1,
                    len([x for x in boundary_points if is_on_boundary(x, coord, limit)]),
                )

    def test_closest_boundary_points_does_not_return_boundary_corners(self):
        """Test that any points equal to boundary corners (up to tolerance) are not
        considered when finding the points on each boundary closest to the collection of
        points."""

        bounds = [(0, 1), (0, 1)]
        domain = SimulatorDomain(bounds)
        inputs = [Input(self.epsilon, 1)]
        self.assertEqual(tuple(), domain.closest_boundary_points(inputs))

        inputs = [Input(self.epsilon, self.epsilon), Input(0.9, 0.8)]
        boundary_points = domain.closest_boundary_points(inputs)
        expected = (
            Input(0, 0.8),
            Input(1, 0.8),
            Input(0.9, 0),
            Input(0.9, 1),
        )
        self.assertTrue(compare_input_tuples(boundary_points, expected))

    def test_calculate_pseudopoints_basic(self):
        """This test checks if the calculate_pseudopoints method is working correctly by
        providing a basic 2D square domain and a collection of points inside it. It validates
        that the method calculates and returns the correct boundary and corner pseudopoints.
        """

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
        self.assertTrue(
            compare_input_tuples(pseudopoints, expected),
            "Pseudopoints calculation is incorrect.",
        )

    def test_calculate_pseudopoints_empty_collection(self):
        """This test ensures that if an empty collection is provided to the
        calculate_pseudopoints method, it returns only the corner pseudopoints."""

        domain = SimulatorDomain([(0, 1), (0, 1)])
        collection = []
        pseudopoints = domain.calculate_pseudopoints(collection)
        expected = (Input(0, 0), Input(0, 1), Input(1, 0), Input(1, 1))
        self.assertTrue(
            compare_input_tuples(pseudopoints, expected),
            "Pseudopoints should only include corner points for empty collection.",
        )

    def test_calculate_pseudopoints_point_outside_domain(self):
        """This test verifies that the calculate_pseudopoints method properly handles points
        outside the domain by raising a ValueError"""

        domain = SimulatorDomain([(0, 1), (0, 1)])
        collection = [Input(0.5, 0.5), Input(1.5, 0.5)]
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

    def test_calculate_pseudopoints_duplicate_points(self):
        """This test ensures that the calculate_pseudopoints method only returns unique points"""

        domain = SimulatorDomain([(0, 1), (0, 1)])
        inputs = [Input(0.5, 0.4), Input(self.epsilon, self.epsilon)]
        pseudopoints = domain.calculate_pseudopoints(inputs)
        expected = (
            Input(0, 0),
            Input(0, 1),
            Input(1, 0),
            Input(1, 1),
            Input(0, 0.4),
            Input(1, 0.4),
            Input(0.5, 0),
            Input(0.5, 1),
        )

        self.assertTrue(compare_input_tuples(pseudopoints, expected))

    def test_get_boundary_mesh_n_type_error(self):
        """This test ensures TypeError is raised if something other than an int is provided 
        for n"""

        domain = SimulatorDomain([(0, 1), (0, 1)])
        n = 2.1
        with self.assertRaisesRegex(
            TypeError, 
            exact(f"Expected 'n' to be of type int, but received {type(n)}."),
        ):
            domain.get_boundary_mesh(n)
        
    def test_get_boundary_mesh_n_value_error(self):
        """This test ensures a ValueError is raised if n !>= 2"""

        domain = SimulatorDomain([(0, 1), (0, 1)])
        n = 1
        with self.assertRaisesRegex(
            ValueError, 
            exact(f"Expected 'n' to be a positive integer >=2 but is equal to {n}."), 
        ):
            domain.get_boundary_mesh(n)
        
    def test_get_boundary_mesh_returns_list_inputs(self):
        """Test that a list of Input objects is returned."""

        domain = SimulatorDomain([(0, 1), (0, 1)])

        for x in domain.get_boundary_mesh(2):
            self.assertIsInstance(x, Input)

    def test_get_boundary_mesh_calculate_values_2d(self):
        """This test ensures the correct boundary values are calculated
        starting with 2D
        """

        domain = SimulatorDomain([(0, 2), (0, 2)])
        n = 3
        mesh_points = domain.get_boundary_mesh(n)
        expected_points = (
            Input(0, 0),
            Input(0, 1),
            Input(0, 2),
            Input(1, 0),
            Input(1, 2),
            Input(2, 0),
            Input(2, 1),
            Input(2, 2)
        )

        self.assertTrue(compare_input_tuples(mesh_points, expected_points))

    def test_get_boundary_mesh_calculate_values_3d(self):
        """This test ensures the correct boundary values are calculated
        in higher dimensions as well e.g 3D"""

        domain = SimulatorDomain([(0, 2), (0, 2), (0, 2)])
        n = 3
        mesh_points = domain.get_boundary_mesh(n)
        expected_points = (
            Input(0, 0, 0), 
            Input(0, 0, 1), 
            Input(0, 0, 2), 
            Input(0, 1, 0), 
            Input(0, 1, 1), 
            Input(0, 1, 2), 
            Input(0, 2, 0), 
            Input(0, 2, 1), 
            Input(0, 2, 2), 
            Input(1, 0, 0), 
            Input(1, 0, 1), 
            Input(1, 0, 2), 
            Input(1, 1, 0),  
            Input(1, 1, 2), 
            Input(1, 2, 0), 
            Input(1, 2, 1), 
            Input(1, 2, 2), 
            Input(2, 0, 0), 
            Input(2, 0, 1), 
            Input(2, 0, 2), 
            Input(2, 1, 0), 
            Input(2, 1, 1), 
            Input(2, 1, 2), 
            Input(2, 2, 0), 
            Input(2, 2, 1), 
            Input(2, 2, 2)
        )

        self.assertTrue(compare_input_tuples(mesh_points, expected_points))

    def test_get_boundary_mesh_different_bounds(self):
        """This test ensures the correct boundary values are calculated with 
        differing bounds in each dimension.         
        """

        domain = SimulatorDomain([(0, 2), (0, 4)])
        n = 3
        mesh_points = domain.get_boundary_mesh(n)
        expected_points = (
            Input(0, 0),
            Input(0, 2),
            Input(0, 4),
            Input(1, 0),
            Input(1, 4),
            Input(2, 0),
            Input(2, 2),
            Input(2, 4)
        )
        self.assertTrue(compare_input_tuples(mesh_points, expected_points))

class StubClass:
    """A stub class for testing level tagging."""

    def __init__(self, a: int, b=None):
        self.a = a
        self.b = b

    def a_plus(self, x: int) -> int:
        return self.a + x

    def set_b(self, b) -> None:
        self.b = b


class TestLevelTagged(ExauqTestCase):
    def setUp(self) -> None:
        self.a = StubClass(1)

    def test_set_and_get_level(self):
        """An object can be tagged with an integer level and this level can be
        subsequently retrieved. When a tag is set, the original object is modified and a
        reference to this object is returned."""

        level = 99
        a = set_level(self.a, level)
        self.assertEqual(level, get_level(a))
        self.assertIs(a, self.a)

    def test_instance_check(self):
        """An object is tagged with a level precisely when it is an instance of
        LevelTagged."""

        obj = StubClass(1)
        self.assertNotIsInstance(obj, LevelTagged)
        obj = set_level(obj, 2)
        self.assertIsInstance(obj, LevelTagged)
        self.assertIsInstance(obj, StubClass)  # check still an object of type A

    def test_set_level_non_int_error(self):
        """A TypeError is raised if a users tries to set a level that is not an integer."""

        level = "1"
        with self.assertRaisesRegex(
            TypeError,
            exact(f"Expected 'level' to be an integer, but received {type(level)}."),
        ):
            _ = set_level(self.a, level)

    def test_get_level_returns_none_if_no_level(self):
        """A level of None is returned if the object hasn't been tagged with a level."""

        self.assertIsNone(get_level(self.a))

    def test_set_level_attribute_already_present(self):
        """A level cannot be set if it would overwrite an existing attributes."""

        setattr(self.a, LevelTagged.level_attr, "foo")
        with self.assertRaisesRegex(
            ValueError,
            f"Cannot set a level on argument 'obj' with value {self.a} as existing attribute "
            f"'{LevelTagged.level_attr}' would be overwritten.",
        ):
            _ = set_level(self.a, 99)

    def test_remove_level_removes_level(self):
        """An object can have its level removed."""

        # A label that is set can be removed.
        a = set_level(self.a, 99)
        remove_level(a)
        self.assertIsNone(get_level(a))
        self.assertIs(self.a, a)


class TestLevelTaggedOld(ExauqTestCase):
    def test_get_level(self):
        """The level tagged onto the object can be obtained from the `level` attribute."""

        class A:
            def __init__(self, a: int, b=None):
                self.a = a
                self.b = b

            def a_plus(self, x: int) -> int:
                return self.a + x

            def set_b(self, b) -> None:
                self.b = b

        class TaggedA(_LevelTaggedOld, A):
            pass

        level = 1
        tagged = TaggedA(level, 99, "b")
        self.assertEqual(level, tagged.level)

    def test_cannot_modify_level(self):
        """The level cannot be modified once set."""

        tagged = _LevelTaggedOld(1)
        with self.assertRaisesRegex(
            AttributeError, "Cannot modify this instance's 'level' attribute."
        ):
            tagged.level = 99

    def test_parent_class_has_level_attribute_error(self):
        """A TypeError is raised if subclassing from LevelTagged would involve masking a
        `level` attribute from a parent class (as worked out from method resolution
        order)."""

        class A:
            def level(self):
                return "level from A"

        with self.assertRaisesRegex(TypeError, "^Cannot create class"):

            class TaggedA(_LevelTaggedOld, A):
                pass

        class B:
            def __init__(self):
                self.level = 99

        with self.assertRaisesRegex(TypeError, "^Cannot initialise object"):

            class TaggedB(_LevelTaggedOld, B):
                pass

            _ = TaggedB(level=10)


class TestMultiLevel(ExauqTestCase):
    def setUp(self) -> None:
        self.elements = ["a", "b", "c"]

    def test_initialise_from_mapping_with_int_keys(self):
        """A MultiLevel collection can be initialised from a mapping with
        integer keys. If keys are not integers then a ValueError is raised. The resulting
        object derives from `dict`."""

        d = dict(zip([2, 4, 5], self.elements))
        ml = MultiLevel(d)
        self.assertIsInstance(ml, dict)

        key = "a"
        with self.assertRaisesRegex(
            ValueError,
            exact(
                f"Key '{key}' of invalid type {type(key)} found: keys should be integers that define levels."
            ),
        ):
            _ = MultiLevel({key: 1})

    def test_initialise_from_sequence(self):
        """If a sequence is supplied, then a multi-level collection is initialised with
        levels starting at 1 and going up to the number of elements in the sequence. The
        values are the corresponding elements of the sequence.
        """

        elements = "abcde"
        levels = tuple(i + 1 for i in range(len(elements)))
        expected = MultiLevel(dict(zip(levels, elements)))
        self.assertEqual(expected, MultiLevel(elements))

    def test_initialise_from_non_sequence_error(self):
        """A TypeError is raised if something other than a Mapping or a sequence is
        supplied for the elements at initialisation."""

        for elems in [{"a", "b"}, zip([1, 2, 3], self.elements)]:
            with self.assertRaisesRegex(
                TypeError,
                exact(
                    "Argument 'elements' must be a mapping with int keys or a sequence, "
                    f"but received object of type {type(elems)}."
                ),
            ):
                _ = MultiLevel(elems)

    def test_equals_detects_class(self):
        """Two multi-level collections are equal if they are both instances of
        MultiLevel and are equal as dicts."""

        d_dict = dict(zip([1, 2, 3], self.elements))
        d1 = MultiLevel(d_dict)
        self.assertNotEqual(d1, d_dict)

        d2 = MultiLevel(d_dict)
        self.assertEqual(d1, d2)

    def test_levels(self):
        """The levels attribute returns the levels as an ordered tuple of ints."""

        self.assertEqual((1, 3, 4), MultiLevel({3: "a", 1: "a", 4: "a"}).levels)
        self.assertEqual((1, 2, 4), MultiLevel({2: "a", 4: "a", 1: "a"}).levels)

    def test_map_preserves_level_structure(self):
        """The map method applies a function to the objects at each level and
        returns the result as a multi-level collection, with the results reflecting
        the same level structure."""

        def f(level: int, x: str) -> str:
            return str(level) + "_" + x

        levels = [2, 4, 6]
        d = dict(zip(levels, self.elements))
        ml = MultiLevel(d)
        expected = MultiLevel(
            {level: f(level, x) for level, x in zip(levels, self.elements)}
        )
        self.assertEqual(expected, ml.map(f))


class TestMultiLevelGaussianProcess(ExauqTestCase):
    def setUp(self) -> None:
        self.gps = {1: WhiteNoiseGP(), 2: WhiteNoiseGP(), 3: WhiteNoiseGP()}
        self.training_data = MultiLevel(
            {
                1: (TrainingDatum(Input(0), 1),),
                2: (TrainingDatum(Input(0.5), 2),),
                3: (TrainingDatum(Input(1), 3),),
            }
        )
        self.hyperparameters = MultiLevel(
            {
                level: WhiteNoiseGPHyperparameters(process_var=level)
                for level in self.training_data
            }
        )
        self.hyperparameter_bounds = MultiLevel(
            {
                level: self.make_white_noise_gp_hyperparameter_bounds((level + 10, None))
                for level in self.training_data
            }
        )

    @staticmethod
    def make_multi_level_gp(
        gps: dict[int, AbstractGaussianProcess],
        coefficients: dict[int, float] = 1,
    ) -> MultiLevelGaussianProcess:
        return MultiLevelGaussianProcess(gps, coefficients=coefficients)

    @staticmethod
    def make_white_noise_gp_hyperparameter_bounds(
        process_var_bounds: OptionalFloatPairs,
    ):
        return [(None, None), process_var_bounds]

    def test_init_type_errors(self):
        """A TypeError is raised if:
        * 'gps' is not a mapping with GPs as values and not a sequence of GPs.
        * 'coefficients' is not a mapping with real number values, not a sequence of
          real numbers and not a real number.
        """

        some_bad_gps = [{1: WhiteNoiseGP, 2: "a"}, [WhiteNoiseGP(), "a"]]
        for bad_gps in some_bad_gps:
            with self.assertRaises(TypeError):
                _ = self.make_multi_level_gp(bad_gps, coefficients=[1, 2])

        some_bad_coeffs = [{1: 1, 2: "a"}, [1, [1]], "a"]
        for bad_coeffs in some_bad_coeffs:
            with self.assertRaises(TypeError):
                _ = self.make_multi_level_gp([WhiteNoiseGP(), WhiteNoiseGP()], bad_coeffs)

    def test_coefficients_single_value_used_at_each_level(self):
        """If a single float is given for the coefficients, then this value is used at
        the coefficient at each level."""

        coefficient = 2
        mlgp = self.make_multi_level_gp(self.gps, coefficients=coefficient)

        expected_coefficients = mlgp.map(lambda level, gp: coefficient)
        self.assertEqual(expected_coefficients, mlgp.coefficients)

    def test_init_from_sequences(self):
        """If a sequence of GPs and a sequence of coefficients are supplied that have the
        same length, then these define the GPs and coefficients at successive levels,
        starting at level 1, in order of the sequences supplied."""

        gps = [WhiteNoiseGP()] * 3
        coefficients = [1, 10, 100]

        mlgp = self.make_multi_level_gp(gps, coefficients)

        levels = [1, 2, 3]
        expected = MultiLevelGaussianProcess(
            gps=dict(zip(levels, gps)), coefficients=dict(zip(levels, coefficients))
        )
        self.assertTrue(all(mlgp[level] == expected[level] for level in levels))
        self.assertEqual(mlgp.coefficients, expected.coefficients)

    def test_init_from_gps_with_different_levels_and_seq_of_coefficients(self):
        """If the GPs have levels different to 1, 2, ... and a sequence of coefficients is
        supplied, then the coefficients are assigned to the GPs in increasing order of
        level."""

        levels = [2, 4, 6]
        gps = {level: WhiteNoiseGP() for level in levels}
        coefficients = [1, 10, 100]

        mlgp = self.make_multi_level_gp(gps, coefficients)

        expected = MultiLevelGaussianProcess(
            gps=gps, coefficients=dict(zip(levels, coefficients))
        )
        self.assertTrue(all(mlgp[level] == expected[level] for level in levels))
        self.assertEqual(mlgp.coefficients, expected.coefficients)

    def test_init_from_sequences_different_lengths_error(self):
        """A ValueError is raised if the lengths of sequences of GPs and coefficients
        differ at initialisation."""

        gps = [WhiteNoiseGP()] * 2
        coefficients = [1, 10, 100]

        with self.assertRaisesRegex(
            ValueError,
            exact(
                "Expected the same number of coefficients as Gaussian processes (got "
                f"{len(coefficients)} coefficients but expected {len(gps)})."
            ),
        ):
            _ = self.make_multi_level_gp(gps, coefficients)

    def test_init_from_mappings_error_gp_level_missing_from_coefficients_levels(self):
        """If multi-level coefficients are supplied at initialisation of a multi-level GP,
        then a ValueError is raised if there is a level from the GPs missing in the levels
        of the coefficients."""

        gps = {1: WhiteNoiseGP(), 3: WhiteNoiseGP()}

        with self.assertRaisesRegex(
            ValueError,
            exact("Missing coefficients for levels: 3."),
        ):
            _ = MultiLevelGaussianProcess(gps, coefficients={1: 1})

        with self.assertRaisesRegex(
            ValueError,
            exact("Missing coefficients for levels: 1, 3."),
        ):
            _ = MultiLevelGaussianProcess(gps, coefficients={2: 2})

    def test_init_from_mappings_ignores_coefficients_for_extra_levels(self):
        """If coefficients are supplied for levels that don't feature in the GPs, then
        these coefficients are ignored."""

        gps = {1: WhiteNoiseGP(), 3: WhiteNoiseGP()}

        mlgp = MultiLevelGaussianProcess(gps, coefficients={1: 1, 2: 2, 3: 3, 4: 4})

        self.assertEqual(MultiLevel({1: 1, 3: 3}), mlgp.coefficients)

    def test_fit_invalid_training_data_error(self):
        """A TypeError is raised if the training data is not an instance of MultiLevel
        or if the objects at each level do not define a valid type for training a
        single-level GP."""

        mlgp = self.make_multi_level_gp({1: MogpEmulator(), 2: MogpEmulator()})
        bad_data = [TrainingDatum(Input(0.5), 1)]
        with self.assertRaisesRegex(
            TypeError,
            exact(
                f"Expected 'training_data' to be an instance of MultiLevel, but received {type(bad_data)}."
            ),
        ):
            mlgp.fit(bad_data)

        with self.assertRaisesRegex(
            TypeError,
            "^Could not train Gaussian process at level 2:",
        ):
            mlgp.fit(MultiLevel({1: [TrainingDatum(Input(0.5), 1)], 2: "a"}))

    def test_fit_invalid_hyperparameters_error(self):
        """An error is raised if the hyperparameters do not define an appropriate object
        for fitting a Gaussian process with."""

        mlgp = self.make_multi_level_gp({1: MogpEmulator(), 2: MogpEmulator()})
        bad_params = "a"
        with self.assertRaisesRegex(
            TypeError,
            "^Could not train Gaussian process at level 1:",
        ):
            mlgp.fit(self.training_data, hyperparameters=bad_params)

    def test_fit_invalid_bounds_error(self):
        """An error is raised if the hyperparameter bounds do not define an appropriate
        object for fitting a Gaussian process with."""

        mlgp = self.make_multi_level_gp({1: MogpEmulator(), 2: MogpEmulator()})
        bad_bounds = 1
        with self.assertRaisesRegex(
            TypeError,
            "^Could not train Gaussian process at level 1:",
        ):
            mlgp.fit(self.training_data, hyperparameter_bounds=bad_bounds)

        bad_bounds = [(None, None), (None, -1)]
        with self.assertRaisesRegex(
            ValueError,
            "^Could not train Gaussian process at level 1:",
        ):
            mlgp.fit(self.training_data, hyperparameter_bounds=bad_bounds)

    def test_fit_fits_data_to_gps_level_wise(self):
        """The constituent GPs within a multi-level GP are fit to Training data
        level-wise."""

        mlgp = self.make_multi_level_gp(self.gps)

        mlgp.fit(self.training_data)

        for level in mlgp.levels:
            self.assertEqual(self.training_data[level], mlgp.training_data[level])

    def test_fit_ignores_extra_levels_in_training_data(self):
        """If the training data has levels that are not in the multi-level GP, then
        the data at those levels is ignored."""

        mlgp = self.make_multi_level_gp(self.gps)
        del mlgp[2]

        mlgp.fit(self.training_data)

        for level in mlgp.levels:
            self.assertEqual(self.training_data[level], mlgp.training_data[level])

    def test_fit_fits_data_to_gps_level_wise_only_on_training_data_levels(self):
        """Only the GPs at levels that appear in the given training data are trained."""

        mlgp = self.make_multi_level_gp(self.gps)
        missing_level = 2
        del self.training_data[missing_level]

        mlgp.fit(self.training_data)

        for level in self.training_data.levels:
            self.assertEqual(
                self.training_data[level],
                mlgp.training_data[level],
            )

        self.assertEqual(tuple(), mlgp.training_data[missing_level])

    def test_fit_multi_level_hyperparameters_applied_level_wise(self):
        """If hyperparameters for multiple levels are supplied, these are applied
        level-wise when fitting the multi-level GP."""

        mlgp = self.make_multi_level_gp(self.gps)

        mlgp.fit(self.training_data, hyperparameters=self.hyperparameters)
        self.assertEqual(self.hyperparameters, mlgp.fit_hyperparameters)

        for level in self.training_data.levels:
            self.assertEqual(self.hyperparameters[level], mlgp[level].fit_hyperparameters)

    def test_fit_single_level_hyperparameters_applied_to_all_levels(self):
        """If a non-levelled set of hyperparameters is supplied, then these are applied to
        each level when fitting the multi-level GP."""

        mlgp = self.make_multi_level_gp(self.gps)
        params = WhiteNoiseGPHyperparameters(process_var=2)
        mlgp.fit(self.training_data, hyperparameters=params)

        for level in mlgp.levels:
            self.assertEqual(params, mlgp.fit_hyperparameters[level])

    def test_fit_default_hyperparameters_for_missing_levels(self):
        """If no hyperparameters are provided for some level of the multi-level GP,
        then the GP at that level has its hyperparameters estimated."""

        noise_level = 11
        gps = {
            1: WhiteNoiseGP(noise_level=noise_level),
            2: WhiteNoiseGP(noise_level=noise_level),
            3: WhiteNoiseGP(noise_level=noise_level),
        }
        mlgp = self.make_multi_level_gp(gps)
        mlparams = MultiLevel(
            {
                1: WhiteNoiseGPHyperparameters(process_var=1),
                3: WhiteNoiseGPHyperparameters(process_var=3),
            }
        )

        mlgp.fit(self.training_data, hyperparameters=mlparams)

        self.assertEqual(
            WhiteNoiseGPHyperparameters(process_var=noise_level),
            mlgp.fit_hyperparameters[2],
        )

    def test_fit_ignores_hyperparameters_at_levels_outside_training_data(self):
        """If the supplied hyperparameters has a level that doesn't feature in the
        training data, then the parameters at this level are ignored."""

        mlgp = self.make_multi_level_gp(self.gps)
        missing_level = 2
        del self.training_data[missing_level]

        mlgp.fit(self.training_data, hyperparameters=self.hyperparameters)

        self.assertIsNone(mlgp.fit_hyperparameters[missing_level])

    def test_fit_multi_level_bounds_applied_level_wise(self):
        """If hyperparameter bounds for multiple levels are supplied, these are applied
        level-wise when fitting the multi-level GP."""

        mlgp = self.make_multi_level_gp(self.gps)

        mlgp.fit(self.training_data, hyperparameter_bounds=self.hyperparameter_bounds)

        # Expected hyperparams: lower bound on process var for each level
        expected_params = MultiLevel(
            {
                level: WhiteNoiseGPHyperparameters(
                    process_var=self.hyperparameter_bounds[level][-1][0]
                )
                for level in self.training_data
            }
        )
        self.assertEqual(expected_params, mlgp.fit_hyperparameters)

    def test_fit_single_level_bounds_applied_to_all_levels(self):
        """If a non-levelled set of hyperparameter bounds is supplied, then these are
        applied to each level when fitting the multi-level GP."""

        mlgp = self.make_multi_level_gp(self.gps)
        lower_bnd = 10
        bounds = self.make_white_noise_gp_hyperparameter_bounds((lower_bnd, None))

        mlgp.fit(self.training_data, hyperparameter_bounds=bounds)

        # Expected hyperparams: lower bound on process var for each level
        for level in mlgp.levels:
            self.assertEqual(
                WhiteNoiseGPHyperparameters(process_var=lower_bnd),
                mlgp.fit_hyperparameters[level],
            )

    def test_fit_default_bounds_for_missing_levels(self):
        """If no hyperparameter bounds are provided for some level of the multi-level GP,
        then the GP at that level has its hyperparameters estimated without any
        constraints applied."""

        noise_level = 11
        gps = {
            1: WhiteNoiseGP(noise_level=noise_level),
            2: WhiteNoiseGP(noise_level=noise_level),
            3: WhiteNoiseGP(noise_level=noise_level),
        }
        mlgp = self.make_multi_level_gp(gps)
        upper_bnd = noise_level - 10
        bounds = MultiLevel(
            {
                1: self.make_white_noise_gp_hyperparameter_bounds((None, upper_bnd)),
                3: self.make_white_noise_gp_hyperparameter_bounds((None, upper_bnd)),
            }
        )

        mlgp.fit(self.training_data, hyperparameter_bounds=bounds)

        self.assertEqual(
            WhiteNoiseGPHyperparameters(process_var=noise_level),
            mlgp.fit_hyperparameters[2],
        )

    def test_fit_ignores_bounds_at_levels_outside_training_data(self):
        """If the supplied hyperparameter bounds have a level that doesn't feature in the
        training data, then the bounds at this level are ignored."""

        mlgp = self.make_multi_level_gp(self.gps)
        missing_level = 2
        del self.training_data[missing_level]

        mlgp.fit(self.training_data, hyperparameter_bounds=self.hyperparameter_bounds)

        self.assertIsNone(mlgp.fit_hyperparameters[missing_level])

    def test_predict_arg_error(self):
        """A TypeError is raised if the input is not an instance of Input."""

        mlgp = self.make_multi_level_gp(self.gps)
        mlgp.fit(self.training_data)
        x = 1
        with self.assertRaisesRegex(
            TypeError,
            exact(
                f"Expected 'x' to be of type {Input.__name__}, but received {type(x)}."
            ),
        ):
            _ = mlgp.predict(x)

    def test_predict_estimate(self):
        """The predicted mean is equal to the sum of the means from the constituent GPs
        multiplied by the corresponding coefficients."""

        coefficients = MultiLevel([1, 10, 100])
        gps = {level: WhiteNoiseGP(prior_mean=-1, noise_level=2) for level in [1, 2, 3]}
        mlgp = self.make_multi_level_gp(gps, coefficients=coefficients)
        mlgp.fit(self.training_data)
        inputs = [
            self.training_data[level][0].input for level in self.training_data.levels
        ] + [
            Input(0.2),
            Input(0.8),
        ]  # add extra points not seen at any of the levels

        for x in inputs:
            level_estimates = mlgp.map(
                lambda level, gp: coefficients[level] * gp.predict(x).estimate
            )
            self.assertEqual(
                sum(level_estimates.values()),
                mlgp.predict(x).estimate,
            )

    def test_predict_variance(self):
        """The predicted variance is equal to the sum of the variances from the
        constituent GPs multiplied by the squares of the corresponding coefficients."""

        coefficients = MultiLevel([1, 10, 100])
        gps = {level: WhiteNoiseGP(prior_mean=-1, noise_level=2) for level in [1, 2, 3]}
        mlgp = self.make_multi_level_gp(gps, coefficients=coefficients)
        mlgp.fit(self.training_data)
        inputs = [
            self.training_data[level][0].input for level in self.training_data.levels
        ] + [
            Input(0.2),
            Input(0.8),
        ]  # add extra points not seen at any of the levels

        for x in inputs:
            level_variances = mlgp.map(
                lambda level, gp: (coefficients[level] ** 2) * gp.predict(x).variance
            )
            self.assertEqual(
                sum(level_variances.values()),
                mlgp.predict(x).variance,
            )


if __name__ == "__main__":
    unittest.main()
