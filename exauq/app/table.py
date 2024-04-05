from collections import OrderedDict
from collections.abc import Sequence
from typing import Any


def make_table(data: OrderedDict[str, Sequence[Any]], formatters=None) -> str:
    # Format contents of cells according to given formatters, or else use string
    # representation
    if formatters is not None:
        formatted_data = OrderedDict(
            [(k, tuple(map(formatters[k], v))) for k, v in data.items()]
        )
    else:
        formatted_data = OrderedDict([(k, tuple(map(str, v))) for k, v in data.items()])

    # Make all cells the same width column-wise
    columns = [[k] + list(v) for k, v in formatted_data.items()]
    max_cell_widths = [max(map(len, col)) for col in columns]
    tidied_columns = []
    for width, column in zip(max_cell_widths, columns):
        fmt = "{" + f":<{width}" + "}"
        tidied_column = [fmt.format(cell) for cell in column]
        tidied_columns.append(tidied_column)

    # Separate cells in rows
    rows = ["  ".join(row_cells) for row_cells in zip(*tidied_columns)]

    # Join rows and return
    return "\n".join(rows)
