import unittest

from tests.utilities.utilities import exact


class TestExact(unittest.TestCase):
    def test_beginning_end_regex(self):
        """Test that the returned regex starts with '^' and ends with '$'."""

        msg = "foo"
        self.assertEqual(f"^{msg}$", exact("foo"))

    def test_parentheses(self):
        """Test that parentheses are escaped in the output regex."""

        self.assertEqual("^\\(foo\\)$", exact("(foo)"))

    def test_backslashes(self):
        """Test that backslashes are escaped in the output regex."""

        self.assertEqual("^windows\\\\path$", exact("windows\\path"))
        self.assertEqual("^\noo$", exact("\noo"))


if __name__ == "__main__":
    unittest.main()
