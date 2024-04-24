import argparse
import json
import pathlib
import re
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
        return self._base_msg


class ExecutionError(Exception):
    """Raised when errors arise from the running of the application."""


class Cli(cmd2.Cmd):
    """The command line interface to the EXAUQ command line application.

    This class implements a command line interpreter using the ``cmd2`` third-party
    package. A 'workspace' directory is used to persist settings relating to a hardware
    interface, the simulator within and a log of simulation jobs submitted.

    Parameters
    ----------
    workspace_dir : exauq.sim_management.types.FilePath
        Path to the workspace directory to use for the app's session.
    """

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

    show_parser = cmd2.Cmd2ArgumentParser()
    show_parser.add_argument(
        "job_ids",
        nargs="*",
        type=str,
        help=(
            "Job IDs to show information for. If not provided, then will show all jobs "
            "subject to the filtering provided by other options."
        ),
    )
    n_jobs_opt_short = "-n"
    n_jobs_opt = "--n-jobs"
    show_parser.add_argument(
        n_jobs_opt_short,
        "--n-jobs",
        nargs="?",
        type=int,
        default=50,
        const=50,
        metavar="N_JOBS",
        help=(
            "the number of jobs to show, counting backwards from the most recently "
            "created (defaults to %(default)s)"
        ),
    )
    show_parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        help=(
            "don't limit the number of jobs to show from the workspace. This "
            f"overrides the {n_jobs_opt_short} argument."
        ),
    )
    status_opt_short = "-s"
    show_parser.add_argument(
        "-s",
        "--status",
        nargs="?",
        default="",
        const="",
        metavar="STATUSES",
        help=(
            "a comma-separated list of statuses, so that only jobs having one of these "
            "statuses will be shown (defaults to '%(default)s', which means show all jobs)"
        ),
    )
    status_not_opt_short = "-S"
    show_parser.add_argument(
        "-S",
        "--status-not",
        nargs="?",
        default="",
        const="",
        metavar="STATUSES",
        help=(
            "a comma-separated list of statuses, so that only jobs *not* having one of these "
            "statuses will be shown (defaults to '%(default)s', which means show all jobs)"
        ),
    )
    result_opt = "--result"
    result_opt_short = "-r"
    show_parser.add_argument(
        "-r",
        result_opt,
        nargs="?",
        choices={"true", "false", ""},
        default="",
        const="true",
        type=lambda x: clean_input_string(x).lower(),
        metavar="true|false",
        help=(
            "whether to show only jobs which have a simulation output ('true'), or show only "
            f"jobs that *don't* have a simulation output ('false'). If {result_opt} is "
            "given as an argument without a value specified, then defaults to '%(const)s'. If "
            f"{result_opt} is not given as an argument, then all jobs will be shown "
            "subject to the other filtering options."
        ),
    )
    show_parser.add_argument(
        "-x",
        "--twr",
        action="store_true",
        help=(
            "show only jobs that have 'terminated without result', i.e. that have no "
            "simulation output yet are no longer running or waiting to run. This overrides "
            "any filters applied to statuses or simulation outputs via with the arguments "
            f"{status_opt_short}, {status_not_opt_short}, and {result_opt_short}."
        ),
    )

    def __init__(self, workspace_dir: FilePath):
        super().__init__(allow_cli_args=False)
        self._workspace_dir = pathlib.Path(workspace_dir)
        self._app = None
        self.prompt = "(exauq)> "
        self._JOBID_HEADER = "JOBID"
        self._INPUT_HEADER = "INPUT"
        self._STATUS_HEADER = "STATUS"
        self._RESULT_HEADER = "RESULT"
        self.table_formatters = {
            self._INPUT_HEADER: format_tuple,
            self._STATUS_HEADER: format_status,
            self._RESULT_HEADER: lambda x: format_float(x, sig_figs=None),
        }
        self.register_preloop_hook(self.initialise_app)

    def initialise_app(self) -> None:
        """Initialise the application with workspace settings.

        The behaviour of this method depends on whether settings can be found in the
        application's workspace directory. If they can, then this implies that
        a workspace was initialised in a previous session of the application, in which
        case this method will initialise a new application instance with the workspace
        settings found. Otherwise, the required settings are gathered from the user and
        stored in the workspace directory, and then a new application instance is
        initialised with these settings.

        Currently the only possible hardware interface that can be used is the
        ``UnixServerScriptInterface``. This methods creates and uses an instance of
        ``UnixServerScriptInterfaceFactory`` to set up this interface.
        """

        general_settings_file = self._workspace_dir / "settings.json"
        hardware_params_file = self._workspace_dir / "hardware_params"
        workspace_log_file = self._workspace_dir / "simulations.csv"

        # TODO: add option to dispatch on plugin
        factory = UnixServerScriptInterfaceFactory()

        if not general_settings_file.exists():
            # Gather settings from UI
            self.poutput(f"A new workspace '{self._workspace_dir}' will be set up.")
            self.poutput(
                "Please provide the following details to initialise the workspace..."
            )
            input_dim = int(input("  Dimension of simulator input space: "))
            for param, prompt in factory.interactive_prompts.items():
                value_str = input(f"  {prompt}: ")
                try:
                    factory.set_param_from_str(param, value_str)
                except ValueError as e:
                    self._render_error(f"Invalid value -- {e}")

            self.poutput("Setting up hardware...")
            hardware = factory.create_hardware()

            # Write settings to file
            self._workspace_dir.mkdir(exist_ok=True)
            write_settings_json(
                {
                    "hardware_type": factory.hardware_cls.__name__,
                    "input_dim": input_dim,
                },
                general_settings_file,
            )
            factory.serialise_hardware_parameters(hardware_params_file)
            self.poutput(f"Thanks -- workspace '{self._workspace_dir}' is now set up.")

            # Create app
            self._app = App(
                interface=hardware,
                input_dim=input_dim,
                simulations_log_file=workspace_log_file,
            )
        else:
            self.poutput(f"Using workspace '{self._workspace_dir}'.")
            general_settings = read_settings_json(general_settings_file)
            factory.load_hardware_parameters(hardware_params_file)
            hardware = factory.create_hardware()
            self._app = App(
                interface=hardware,
                input_dim=general_settings["input_dim"],
                simulations_log_file=workspace_log_file,
            )
        return None

    def do_quit(self, args) -> Optional[bool]:
        """Exit the application."""

        if self._app is not None:
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
        """Write text to standard output."""

        self.poutput(text + "\n")

    def _render_error(self, text: str) -> None:
        """Write text as an error message to standard error."""

        self.perror("Error: " + text)

    def _make_table(self, data: OrderedDict[str, Sequence[Any]]) -> str:
        """Make a textual table from data."""

        return make_table(data, formatters=self.table_formatters)

    def _make_submissions_table(self, jobs: tuple[Job]) -> str:
        """Make table of submitted jobs for displaying to the user."""

        ids = tuple(job.id for job in jobs)
        inputs = tuple(job.data for job in jobs)
        data = OrderedDict([(self._JOBID_HEADER, ids), (self._INPUT_HEADER, inputs)])
        return self._make_table(data)

    @cmd2.with_argparser(submit_parser)
    def do_submit(self, args) -> None:
        """Submit a job to the simulator."""

        try:
            inputs = self._parse_inputs(args.inputs) + self._parse_inputs(args.file)
            submitted_jobs = self._app.submit(inputs)
            self._render_stdout(self._make_submissions_table(submitted_jobs))
        except ParsingError as e:
            self._render_error(str(e))

    def _make_show_table(self, jobs: Sequence[dict[str, Any]]) -> str:
        """Make table of job information for displaying to the user."""

        data = OrderedDict(
            [
                (self._JOBID_HEADER, (job["job_id"] for job in jobs)),
                (self._INPUT_HEADER, (job["input"] for job in jobs)),
                (self._STATUS_HEADER, (job["status"] for job in jobs)),
                (self._RESULT_HEADER, (job["output"] for job in jobs)),
            ]
        )
        return self._make_table(data)

    def _parse_statuses_string_to_set(
        self, statuses: str, empty_to_all: bool = False
    ) -> set[JobStatus]:
        statuses = clean_input_string(statuses)

        # Get comma-separated components
        statuses = statuses.split(",")

        # Remove leading and trailing whitespace, replace inner whitespace with a single
        # underscore, and convert to upper case
        statuses = {re.sub("\\s+", "_", status.strip()).upper() for status in statuses}

        # Return as set of job statuses
        if statuses == {""} and empty_to_all:
            return set(JobStatus)
        else:
            return {x for x in JobStatus if x.name in statuses}

    def _parse_bool(self, result: str) -> Optional[bool]:
        if result == "true":
            return True
        elif result == "false":
            return False
        else:
            return None

    def _parse_show_args(self, args) -> dict[str, Any]:
        if args.n_jobs < 0:
            raise ParsingError(
                f"Value for {self.n_jobs_opt_short}/{self.n_jobs_opt} must be a non-negative integer."
            )

        if args.twr:
            statuses = set(JobStatus) - {
                JobStatus.NOT_SUBMITTED,
                JobStatus.SUBMITTED,
                JobStatus.RUNNING,
            }
            result_filter = False
        else:
            statuses_included = self._parse_statuses_string_to_set(
                args.status, empty_to_all=True
            )
            statuses_excluded = self._parse_statuses_string_to_set(args.status_not)
            statuses = statuses_included - statuses_excluded
            result_filter = self._parse_bool(args.result)

        job_ids = args.job_ids if args.job_ids else None
        n_most_recent = args.n_jobs if not args.all else None
        return {
            "job_ids": job_ids,
            "n_most_recent": n_most_recent,
            "statuses": statuses,
            "result_filter": result_filter,
        }

    @cmd2.with_argparser(show_parser)
    def do_show(self, args) -> None:
        """Show information about jobs."""
        try:
            kwargs = self._parse_show_args(args)
            jobs = self._app.get_jobs(**kwargs)
            self._render_stdout(self._make_show_table(jobs))
        except ParsingError as e:
            self._render_error(str(e))


def clean_input_string(string: str) -> str:
    return string.strip().strip("'\"")


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
    """Serialise a dict of settings to a JSON file.

    Dictionary values should be of types that can be serialised into JSON; note that not
    all Python objects can serialised in this way. (For example, sets cannot be
    serialised.)

    Parameters
    ----------
    settings : dict[str, Any]
        The settings to be serialised.
    path : FilePath
        Path to a text file to write the JSON-serialised settings to.
    """
    with open(path, mode="w") as f:
        json.dump(settings, f, indent=4)


def read_settings_json(path: FilePath) -> dict[str, Any]:
    """Read settings from a JSON file.

    Deserialises the JSON data into a Python object.

    Parameters
    ----------
    path : FilePath
        Path to a JSON file of the settings.

    Returns
    -------
    dict[str, Any]
        The settings stored in the JSON file.
    """
    with open(path, mode="r") as f:
        return json.load(f)
