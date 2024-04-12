import argparse
import json
import pathlib
import sys
from collections import OrderedDict
from collections.abc import Sequence
from io import TextIOWrapper
from typing import Any, Callable, Optional, Union

import cmd2

from exauq.app.app import App
from exauq.app.startup import UnixServerScriptInterfaceFactory
from exauq.sim_management.hardware import JobStatus
from exauq.sim_management.jobs import Job
from exauq.sim_management.types import FilePath


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
        super().__init__(allow_cli_args=False)
        self._app = app
        self.prompt = "(exauq)> "
        self.JOBID_HEADER = "JOBID"
        self.INPUT_HEADER = "INPUT"
        self.STATUS_HEADER = "STATUS"
        self.RESULT_HEADER = "RESULT"
        self.table_formatters = {
            self.INPUT_HEADER: format_tuple,
            self.STATUS_HEADER: format_status,
            self.RESULT_HEADER: lambda x: format_float(x, sig_figs=None),
        }

    def do_quit(self, args) -> Optional[bool]:
        """Exit the application."""

        self._app.shutdown()
        return super().do_quit(args)

    def _parse_inputs(
        self, inputs: Union[Sequence[str], TextIOWrapper]
    ) -> list[tuple[float, ...]]:
        """Convert string representations of simulator inputs to tuples of floats."""

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
        """Write text to the application's standard output."""

        self.poutput(text + "\n")

    def _make_table(self, data: OrderedDict[str, Sequence[Any]]) -> str:
        """Make a textual table from data."""

        return make_table(data, formatters=self.table_formatters)

    def _make_submissions_table(self, jobs: tuple[Job]) -> str:
        """Make table of submitted jobs for displaying to the user."""

        ids = tuple(job.id for job in jobs)
        inputs = tuple(job.data for job in jobs)
        data = OrderedDict([(self.JOBID_HEADER, ids), (self.INPUT_HEADER, inputs)])
        return self._make_table(data)

    @cmd2.with_argparser(submit_parser)
    def do_submit(self, args) -> None:
        """Submit a job to the simulator."""

        try:
            inputs = self._parse_inputs(args.inputs) + self._parse_inputs(args.file)
            submitted_jobs = self._app.submit(inputs)
            self._render_stdout(self._make_submissions_table(submitted_jobs))
        except ParsingError as e:
            self.perror(str(e))

    def _make_show_table(self, jobs: Sequence[dict[str, Any]]) -> str:
        """Make table of job information for displaying to the user."""

        data = OrderedDict(
            [
                (self.JOBID_HEADER, (job["job_id"] for job in jobs)),
                (self.INPUT_HEADER, (job["input"] for job in jobs)),
                (self.STATUS_HEADER, (job["status"] for job in jobs)),
                (self.RESULT_HEADER, (job["output"] for job in jobs)),
            ]
        )
        return self._make_table(data)

    def do_show(self, args) -> None:
        """Show information about jobs."""

        jobs = self._app.get_jobs()
        self._render_stdout(self._make_show_table(jobs))


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


def format_float(x: Union[float, None], sig_figs: Optional[int] = None) -> str:
    """Format floats to a specified number of significant figures.

    Parameters
    ----------
    x : float or None
        A floating point number. If ``None`` then an empty string is returned.
    sig_figs : int, optional
        (Default: None) The number of significant figures to display. If ``None``
        then display with full precision.

    Returns
    -------
    str
        The floating point number rounded to the given number of significant figures,
        or the empty string if ``None`` was provided for `x`.
    """
    if x is None:
        return ""
    else:
        fmt = "{" + f":.{sig_figs}" + "}" if sig_figs is not None else "{:}"
        return fmt.format(x)


def format_tuple(x: tuple[float]) -> str:
    """Format a tuple of floats to 3 significant figures.

    Parameters
    ----------
    x : tuple[float]
        A tuple of floating point numbers.

    Returns
    -------
    str
        The tuple, with each floating point number within rounded to 3 significant
        figures.

    Examples
    --------
    >>> format_tuple((1.1111, 9.9999, 3.1))
    '(1.11, 10.0, 3.1)'
    """

    return "(" + ", ".join([format_float(flt, sig_figs=3) for flt in x]) + ")"


def format_status(status: JobStatus) -> str:
    """Format a job status, returning the string value of the enum member.

    Parameters
    ----------
    status : JobStatus
        A job status.

    Returns
    -------
    str
        The enum value of the status.
    """
    return str(status.value)


def write_settings_json(settings: dict[str, Any], path: FilePath) -> None:
    with open(path, mode="w") as f:
        json.dump(settings, f, indent=4)


def read_settings_json(path: FilePath) -> dict[str, dict[str, Any]]:
    with open(path, mode="r") as f:
        return json.load(f)


def main():
    workspace_dir = pathlib.Path(".exauq-ws")
    general_settings_file = workspace_dir / "settings.json"
    hardware_params_file = workspace_dir / "hardware_params"
    workspace_log_file = workspace_dir / "simulations.csv"
    factory = UnixServerScriptInterfaceFactory()

    if not general_settings_file.exists():
        # Gather settings from UI
        print(f"A new workspace '{workspace_dir}' will be set up.")
        print("Please provide the following details to initialise the workspace...")
        input_dim = int(input("Dimension of simulator input space: "))
        hardware = factory.make_hardware_interactively()

        # Write settings to file
        workspace_dir.mkdir(exist_ok=True)
        write_settings_json(
            {
                "hardware_type": factory.hardware_type,
                "input_dim": input_dim,
            },
            general_settings_file,
        )
        factory.serialise_hardware_parameters(hardware_params_file)
        print(f"Thanks. '{workspace_dir}' is now set up.")

        # Create app
        app = App(
            interface=hardware,
            input_dim=input_dim,
            simulations_log_file=workspace_log_file,
        )
    else:
        print(f"Using workspace '{workspace_dir}'.")
        general_settings = read_settings_json(general_settings_file)
        hardware = factory.load_hardware(hardware_params_file)
        app = App(
            interface=hardware,
            input_dim=general_settings["input_dim"],
            simulations_log_file=workspace_log_file,
        )

    # Make CLI app and run
    cli = Cli(app)
    sys.exit(cli.cmdloop())


if __name__ == "__main__":
    main()
