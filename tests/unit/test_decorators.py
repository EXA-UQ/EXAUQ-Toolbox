import contextlib
import io
import unittest

from exauq.utilities.decorators import suppress_print


class TestSuppressPrint(unittest.TestCase):
    def test_suppress_print(self):
        """Test that the suppress_print decorator suppresses print statements."""

        @suppress_print
        def test_print():
            print("This is a test print statement")

        with contextlib.redirect_stdout(io.StringIO()) as new_stdout:
            test_print()
            self.assertEqual(new_stdout.getvalue(), "")

    def test_return_value(self):
        """Test that the suppress_print decorator does not suppress return values."""

        @suppress_print
        def test_print():
            return 5

        self.assertEqual(test_print(), 5)

    def test_suppress_print_with_return_value(self):
        """Check that the decorator suppresses print statements but does not interfere
        with return values."""

        @suppress_print
        def function_with_return_value():
            print("This print should be suppressed")
            return 42

        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            result = function_with_return_value()

        self.assertEqual(stdout.getvalue(), "")

        self.assertEqual(result, 42)

    def test_suppress_print_no_print(self):
        """Test that the suppress_print decorator does not affect functions that do
        not print."""

        @suppress_print
        def no_print_function():
            return "No print here"

        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            result = no_print_function()

        self.assertEqual(stdout.getvalue(), "")

        self.assertEqual(result, "No print here")

    def test_suppress_print_multiple_prints(self):
        """Test that the suppress_print decorator suppresses multiple print statements."""

        @suppress_print
        def multiple_prints():
            print("First print")
            print("Second print")
            print("Third print")
            return "Done"

        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            result = multiple_prints()

        self.assertEqual(stdout.getvalue(), "")

        self.assertEqual(result, "Done")

    def test_suppress_print_with_exception(self):
        """Check that the decorator does not suppress exception messages, and that
        exceptions raised within the function are still handled properly."""

        @suppress_print
        def function_with_exception():
            print("This will be suppressed")
            raise ValueError("This is an exception")

        with self.assertRaises(ValueError) as context:
            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                function_with_exception()

        self.assertEqual(stdout.getvalue(), "")

        self.assertEqual(str(context.exception), "This is an exception")

    def test_suppress_print_with_kwargs(self):
        """Verify that the decorator works correctly on functions that take keyword
        arguments."""

        @suppress_print
        def function_with_kwargs(value="default"):
            print(f"Print with value: {value}")
            return value

        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            result = function_with_kwargs(value="test")

        self.assertEqual(stdout.getvalue(), "")

        self.assertEqual(result, "test")

    def test_suppress_print_with_args_kwargs(self):
        """Ensure the decorator handles both positional and keyword arguments without
        issues."""

        @suppress_print
        def function_with_args_kwargs(a, b, c="default"):
            print(f"Positional: {a}, {b}, Keyword: {c}")
            return a + b

        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            result = function_with_args_kwargs(1, 2, c="keyword")

        self.assertEqual(stdout.getvalue(), "")

        self.assertEqual(result, 3)
