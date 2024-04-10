import argparse
import sys
from collections import OrderedDict
from collections.abc import Sequence
from io import TextIOWrapper
from typing import Any, Callable, Union

import cmd2

from exauq.app.app import App
from exauq.sim_management.hardware import UnixServerScriptInterface
from exauq.sim_management.jobs import Job


class ParsingError(Exception):
    """Raised when errors arise from parsing command line arguments."""

    def __init__(self, e: Union[str, Exception]):
        self._base_msg = str(e)

    def __str__(self):
        return f"Error parsing args: {self._base_msg}"


class ExecutionError(Exception):
    """Raised when errors arise from the running of the application."""


class Cli(cmd2.Cmd):
    """The command line interface to the exauq application."""

    JOBID_HEADER = "JOBID"
    INPUTS_HEADER = "INPUTS"

    submit_parser = cmd2.Cmd2ArgumentParser()
    submit_parser.add_argument(
        "inputs",
        nargs="*",
        type=str,
        help="The inputs to submit to the simulator.",
    )
    submit_parser.add_argument(
        "-f",
        "--file",
        type=argparse.FileType(mode="r"),
        help="A path to a csv file containing inputs to submit to the simulator.",
    )

    def __init__(self, app: App):
        super().__init__()
        self._app = app
        self.prompt = "(exauq)> "

    def _parse_inputs(
        self, inputs: Union[Sequence[str], TextIOWrapper]
    ) -> list[tuple[float, ...]]:
        if inputs:
            try:
                return [tuple(map(float, x.split(","))) for x in inputs]
            except ValueError as e:
                raise ParsingError(e)
            finally:
                if isinstance(inputs, TextIOWrapper):
                    inputs.close()
        else:
            return []

    def _render_stdout(self, text: str) -> None:
        self.poutput(text + "\n")

    def _make_submissions_table(self, jobs: tuple[Job]) -> str:
        ids = tuple(job.id for job in jobs)
        inputs = tuple(job.data for job in jobs)
        data = OrderedDict([(self.JOBID_HEADER, ids), (self.INPUTS_HEADER, inputs)])
        return make_table(data, formatters={self.INPUTS_HEADER: format_tuple})

    @cmd2.with_argparser(submit_parser)
    def do_submit(self, args) -> None:
        """Submit a job to the simulator."""

        try:
            inputs = self._parse_inputs(args.inputs) + self._parse_inputs(args.file)
            submitted_jobs = self._app.submit(inputs)
            self._render_stdout(self._make_submissions_table(submitted_jobs))
        except ParsingError as e:
            self.perror(str(e))

    def do_status(self, args) -> None:
        """Get the status of simulation jobs."""

        _ = self._app.status()
        self.poutput("** Rendering of statuses. **")

    def do_result(self, args) -> None:
        """Retrieve the result of simulation jobs"""

        _ = self._app.result()
        self.poutput("** Rendering of results. **")


def make_table(
    data: OrderedDict[str, Sequence[Any]],
    formatters: dict[str, Callable[[Any], str]] = None,
) -> str:
    """Make a table of data as a string.

    Each column in the table is left-justified and columns are separated with two spaces.
    The contents of each data cell in a column is formatted according to the supplied
    formatting function, if any, or else just as a string using Python's built-in ``str``
    function. The width of the column is equal to the length of the longest (string
    formatted) cell value in the column (including the column header).

    Parameters
    ----------
    data : OrderedDict[str, Sequence[Any]]
        The data to put into the table. The keys of the ordered dict should be the
        table column headers (formatted as desired for the table) and the values should
        be the values in the columns. The order of the columns output is given by the
        order of the corresponding keys in the ordered dict.
    formatters : dict[str, Callable[[Any], str]], optional
        (Default: None) A collection of formatting functions to apply to columns. Keys of
        the dict should be column headings. The value for a column heading should be a
        single argument function that can be applied to a value in the column and return a
        string representation of that value. If `None` then all columns will be converted
        to strings using Python's ``str`` built-in function.

    Returns
    -------
    str
        A table of the data, with columns formatted according to the supplied formatters.
    """

    # Format contents of table cells according to given formatters, or else use string
    # representation
    if formatters is not None:
        formatters |= {k: str for k in data if k not in formatters}
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

    # Separate cells in rows with two spaces
    rows = ["  ".join(row_cells) for row_cells in zip(*tidied_columns)]

    # Join rows to return a single string
    return "\n".join(rows)


def format_float(x: float, dp: int = 2) -> str:
    """Format floats to a specified number of decimal places.

    Parameters
    ----------
    x : float
        A floating point number.
    dp : int, optional
        (Default: 2) The number of decimal places to round to.

    Returns
    -------
    str
        The rounded value of the given number.
    """

    fmt = "{" + f":.{dp}f" + "}"
    return fmt.format(x)


def format_tuple(x: tuple[float]) -> str:
    """Format a tuple of floats to 2 decimal places.

    Parameters
    ----------
    x : tuple[float]
        A tuple of floating point numbers.

    Returns
    -------
    str
        The tuple, with each floating point number within rounded to 2 decimal places.

    Examples
    --------
    >>> format_tuple((1.1111, 9.9999, 3.1))
    '(1.11, 10.00, 3.10)'
    """

    return "(" + ", ".join([format_float(flt, dp=2) for flt in x]) + ")"


def main():
    cli = Cli(App())
    sys.exit(cli.cmdloop())


if __name__ == "__main__":
    main()
