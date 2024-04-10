import unittest
from collections import OrderedDict
from textwrap import dedent

from exauq.app.cli import make_table


class TestMakeTable(unittest.TestCase):
    def test_empty_table_shows_header(self):
        """The header row is shown even if there is no data."""

        data = OrderedDict(
            [
                ("COL1", []),
                ("COL2", []),
            ]
        )
        self.assertEqual("COL1  COL2", make_table(data))

    def test_cells_separated_by_two_spaces(self):
        """Each cell in the table is separated by at least two spaces."""

        data = OrderedDict(
            [
                ("COL1", ["aaaa"]),
                ("COL2", ["bbbb"]),
            ]
        )
        self.assertEqual("COL1  COL2\naaaa  bbbb", make_table(data))

    def test_all_cells_in_columns_have_same_width(self):
        """For each column, the cells within the column have the same width."""

        data = OrderedDict(
            [
                ("C1", ["aaa"]),
                ("COL2", ["bbb"]),
            ]
        )
        self.assertEqual("C1   COL2\naaa  bbb ", make_table(data))

    def test_apply_custom_string_representations_to_columns(self):
        """The values in columns can be represented as strings with custom functions,
        which may differ by column."""

        def fmt_float(x: float) -> str:
            """Format floats to 2dp."""
            return f"{x:.2f}"

        def fmt_tuple(x: tuple[float]) -> str:
            """Format tuple of floats as comma-separated floats to 2dp."""
            return "(" + ", ".join(map(fmt_float, x)) + ")"

        data = OrderedDict(
            [
                ("COL1", [1.111111, 9.99999]),
                ("COL2", [(1.11111, 2.22222), (5.55555, 6.66666)]),
            ]
        )
        formatters = {"COL1": fmt_float, "COL2": fmt_tuple}
        expected = dedent(
            """
            COL1   COL2        
            1.11   (1.11, 2.22)
            10.00  (5.56, 6.67)
            """
        ).strip()
        self.assertEqual(expected, make_table(data, formatters=formatters))


if __name__ == "__main__":
    unittest.main()
