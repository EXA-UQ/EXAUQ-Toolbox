import argparse
import csv
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
from exauq.sim_management.jobs import Job, JobId
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

    INTERFACE_FACTORIES = {
        "1": ("Unix Server Script Interface", UnixServerScriptInterfaceFactory),
        # "2": ("Windows Server Script Interface", WindowsServerScriptInterfaceFactory),
        # Add more factories if needed
    }

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
        type=argparse.FileType(mode="r", encoding="utf-8-sig"),
        help="A path to a csv file containing inputs to submit to the simulator.",
    )

    resubmit_parser = cmd2.Cmd2ArgumentParser()
    resubmit_parser.add_argument(
        "job_ids",
        nargs="*",
        type=str,
        help="Job IDs of the jobs to resubmit.",
    )
    status_opt_short = "-s"
    resubmit_parser.add_argument(
        status_opt_short,
        "--status",
        nargs="?",
        default="",
        const="",
        metavar="STATUSES",
        help=(
            "a comma-separated list of statuses, so that only jobs having one of these "
            "statuses will be resubmitted (defaults to '%(default)s', which means resubmit all jobs)"
        ),
    )
    status_not_opt_short = "-S"
    resubmit_parser.add_argument(
        status_not_opt_short,
        "--status-not",
        nargs="?",
        default="",
        const="",
        metavar="STATUSES",
        help=(
            "a comma-separated list of statuses, so that only jobs *not* having one of these "
            "statuses will be resubmitted (defaults to '%(default)s', which means resubmit all jobs)"
        ),
    )
    resubmit_parser.add_argument(
        "-x",
        "--twr",
        action="store_true",
        help=(
            "Resubmit all jobs that have 'terminated without result'."
            "This overrides any filters applied to statuses with the arguments "
            f"{status_opt_short} and {status_not_opt_short}."
        ),
    )

    cancel_parser = cmd2.Cmd2ArgumentParser()
    cancel_parser.add_argument(
        "job_ids",
        nargs="+",
        type=str,
        help=("Job IDs of the jobs to cancel."),
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
    show_parser.add_argument(
        status_opt_short,
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
    show_parser.add_argument(
        status_not_opt_short,
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

    write_parser = cmd2.Cmd2ArgumentParser()
    write_parser.add_argument(
        "file",
        type=argparse.FileType(mode="w"),
        help="A path to a csv file to write job details to.",
    )

    add_interface_parser = cmd2.Cmd2ArgumentParser()
    add_interface_parser.add_argument(
        "file",
        type=argparse.FileType(mode="r"),
        help="Path to the JSON file for the hardware interface to add to the workspace.",
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
        self._HEADER_MAPPER = {
            "job_id": self._JOBID_HEADER,
            "input": self._INPUT_HEADER,
            "status": self._STATUS_HEADER,
            "output": self._RESULT_HEADER,
        }
        self.table_formatters = {
            self._INPUT_HEADER: format_tuple,
            self._STATUS_HEADER: format_status,
            self._RESULT_HEADER: lambda x: format_float(x, sig_figs=None),
        }
        self.register_preloop_hook(self.initialise_app)
        self._hardware_params_prefix = "hw_params_"

    def initialise_app(self) -> None:
        """Initialise the application with workspace settings.

        The behaviour of this method depends on whether settings can be found in the
        application's workspace directory. If they can, then this implies that
        a workspace was initialised in a previous session of the application, in which
        case this method will initialise a new application instance with the workspace
        settings found. Otherwise, the required settings are gathered from the user and
        stored in the workspace directory, and then a new application instance is
        initialised with these settings.

        Currently, the only possible hardware interface that can be used is the
        ``UnixServerScriptInterface``. This method creates and uses an instance of
        ``UnixServerScriptInterfaceFactory`` to set up this interface.
        """

        general_settings_file = self._workspace_dir / "settings.json"
        workspace_log_file = self._workspace_dir / "simulations.csv"

        hardware_interfaces = []
        interface_details = {}

        # TODO: add option to dispatch on plugin

        if not general_settings_file.exists():
            # Gather settings from UI
            self.poutput(f"A new workspace '{self._workspace_dir}' will be set up.")
            self.poutput(
                "Please provide the following details to initialise the workspace..."
            )
            input_dim = int(input("  Dimension of simulator input space: "))

            while True:
                # Display factory options
                self.poutput("Select the hardware interface type you wish to use:")
                for option, (display_name, _) in Cli.INTERFACE_FACTORIES.items():
                    self.poutput(f"  {option}: {display_name}")

                factory_choice = input("Enter the number corresponding to your choice: ")
                selected_factory = Cli.INTERFACE_FACTORIES.get(factory_choice)

                if not selected_factory:
                    self.poutput("Invalid choice, please try again.")
                    continue

                display_name, factory_cls = selected_factory
                factory = factory_cls()

                self.poutput(f"Selected: {display_name}")

                self.poutput(
                    "Please provide the following details for your hardware "
                    "interface..."
                )
                for param, prompt in factory.interactive_prompts.items():
                    value_str = input(f"  {prompt}: ")
                    try:
                        factory.set_param_from_str(param, value_str)
                    except ValueError as e:
                        self._render_error(f"Invalid value -- {e}")

                self.poutput("Setting up hardware...")
                hardware_interfaces.append(factory.create_hardware())

                self._workspace_dir.mkdir(exist_ok=True)
                interface_name = hardware_interfaces[-1].name
                hardware_params_filename = (
                    self._hardware_params_prefix + interface_name + ".json"
                )
                hardware_params_file = self._workspace_dir / hardware_params_filename
                factory.serialise_hardware_parameters(hardware_params_file)

                interface_details[interface_name] = {
                    "factory": factory_cls.__name__,
                    "params": hardware_params_filename,
                }

                if input("  Add another hardware interface? (y/n): ").lower() != "y":
                    break

            # Write settings to file
            write_settings_json(
                {
                    "interfaces": interface_details,
                    "input_dim": input_dim,
                },
                general_settings_file,
            )

            self.poutput(f"Thanks -- workspace '{self._workspace_dir}' is now set up.")

            # Create app
            self._app = App(
                interfaces=hardware_interfaces,
                input_dim=input_dim,
                simulations_log_file=workspace_log_file,
            )
        else:
            self.poutput(f"Using workspace '{self._workspace_dir}'.")
            general_settings = read_settings_json(general_settings_file)

            for interface_name, interface_details in general_settings[
                "interfaces"
            ].items():
                factory_cls = interface_details["factory"]
                hardware_params_filename = interface_details["params"]
                hardware_params_file = self._workspace_dir / hardware_params_filename
                factory = globals()[factory_cls]()

                factory.load_hardware_parameters(hardware_params_file)
                hardware_interfaces.append(factory.create_hardware())

            self._app = App(
                interfaces=hardware_interfaces,
                input_dim=general_settings["input_dim"],
                simulations_log_file=workspace_log_file,
            )
        return None

    def do_quit(self, args) -> Optional[bool]:
        """Exit the application."""

        if self._app is not None:
            self._app.shutdown()

        return super().do_quit(args)

    def _render_stdout(self, text: str, trailing_newline: bool = True) -> None:
        """Write text to standard output, with an optional trailing newline character."""

        if trailing_newline:
            self.poutput(text + "\n")
        else:
            self.poutput(text)

    def _render_error(self, text: str) -> None:
        """Write text as an error message to standard error."""

        self.perror("Error: " + text)

    def _render_warning(self, text: str) -> None:
        """Write text as a warning message to standard error."""

        self.pwarning("Warning: " + text)

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
        """Submit jobs to the simulator."""

        try:
            inputs = parse_inputs(args.inputs) + parse_inputs(args.file)
            submitted_jobs = self._app.submit(inputs)
            self._render_stdout(self._make_submissions_table(submitted_jobs))
        except ParsingError as e:
            self._render_error(str(e))
        finally:
            if isinstance(args.file, TextIOWrapper):
                args.file.close()

    def _parse_resubmit_args(self, args) -> dict[str, Any]:
        """Convert command line arguments for the resubmit command to a dict of arguments for
        the application to process.
        """
        if args.twr:
            statuses = {JobStatus.CANCELLED, JobStatus.FAILED, JobStatus.FAILED_SUBMIT}
        else:
            statuses_included = parse_statuses_string_to_set(
                args.status, empty_to_all=True
            )
            statuses_excluded = parse_statuses_string_to_set(args.status_not)
            statuses = statuses_included - statuses_excluded

        job_ids = parse_job_ids(args.job_ids) if args.job_ids else None

        return {
            "job_ids": job_ids,
            "statuses": statuses,
        }

    def _make_resubmissions_table(self, jobs: list[tuple[Any, Any, Any]]) -> str:
        """Make table of resubmitted jobs for displaying to the user."""

        old_ids = tuple(old_id for old_id, _, _ in jobs)
        new_ids = tuple(new_id for _, new_id, _ in jobs)
        inputs = tuple(data for _, _, data in jobs)
        data = OrderedDict(
            [("OLD_JOBID", old_ids), ("NEW_JOBID", new_ids), (self._INPUT_HEADER, inputs)]
        )
        return self._make_table(data)

    @cmd2.with_argparser(resubmit_parser)
    def do_resubmit(self, args) -> None:
        """Resubmit jobs to the simulator."""

        try:
            kwargs = self._parse_resubmit_args(args)
            jobs = self._app.get_jobs(**kwargs)

            resubmitted_jobs = []
            for job in jobs:
                new_job = self._app.submit([job["input"]])[0]
                resubmitted_jobs.append((job["job_id"], new_job.id, job["input"]))

            self._render_stdout(self._make_resubmissions_table(resubmitted_jobs))
        except ParsingError as e:
            self._render_error(str(e))

    def _make_cancel_table(self, jobs: Sequence[dict[str, Any]]) -> str:
        """Make table of details of cancelled jobs for displaying to the user."""

        col_headings = {
            "job_id": self._JOBID_HEADER,
            "input": self._INPUT_HEADER,
            "status": self._STATUS_HEADER,
        }
        data = OrderedDict(
            [
                (header, tuple(job[k] for job in jobs))
                for k, header in col_headings.items()
            ]
        )

        return self._make_table(data)

    @cmd2.with_argparser(cancel_parser)
    def do_cancel(self, args) -> None:
        "Cancel simulation jobs."

        try:
            job_ids = parse_job_ids(args.job_ids)
        except ParsingError as e:
            self._render_error(str(e))
            return None

        report = self._app.cancel(job_ids)
        cancelled_jobs = report["cancelled_jobs"]
        if cancelled_jobs:
            self._render_stdout(self._make_cancel_table(cancelled_jobs))

        terminated_jobs = [str(job_id) for job_id in report["terminated_jobs"]]
        if terminated_jobs:
            self._render_stdout(
                "The following jobs have already terminated and were not cancelled:\n"
                + "\n".join(f"  {job_id}" for job_id in terminated_jobs),
                trailing_newline=False,
            )

        non_existent_jobs = [str(job_id) for job_id in report["non_existent_jobs"]]
        if non_existent_jobs:
            self._render_warning(
                "Could not find jobs with the following IDs:\n"
                + "\n".join(f"  {job_id}" for job_id in non_existent_jobs)
            )

    def _make_show_table(self, jobs: Sequence[dict[str, Any]]) -> str:
        """Make table of job information for displaying to the user."""

        data = OrderedDict(
            [
                (header, tuple(job[k] for job in jobs))
                for k, header in self._HEADER_MAPPER.items()
            ]
        )

        return self._make_table(data)

    def _parse_show_args(self, args) -> dict[str, Any]:
        """Convert command line arguments for the show command to a dict of arguments for
        the application to process.
        """
        if args.n_jobs < 0:
            raise ParsingError(
                f"Value for {self.n_jobs_opt_short}/{self.n_jobs_opt} must be a non-negative integer."
            )

        if args.twr:
            statuses = set(JobStatus) - {
                JobStatus.PENDING_SUBMIT,
                JobStatus.SUBMITTED,
                JobStatus.RUNNING,
            }
            result_filter = False
        else:
            statuses_included = parse_statuses_string_to_set(
                args.status, empty_to_all=True
            )
            statuses_excluded = parse_statuses_string_to_set(args.status_not)
            statuses = statuses_included - statuses_excluded
            result_filter = parse_bool(args.result)

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

    @cmd2.with_argparser(write_parser)
    def do_write(self, args) -> None:
        """Write details of jobs in this workspace to a CSV file."""

        jobs = [
            self._restructure_record_for_csv(record) for record in self._app.get_jobs()
        ]
        try:
            self._write_to_csv(jobs, args.file)
        except Exception as e:
            self._render_error(str(e))
        finally:
            args.file.close()

    def _restructure_record_for_csv(self, job_record: dict[str, Any]) -> dict[str, Any]:
        """Convert job information to a dict that's suitable for writing to a CSV.

        Performs the following steps:

        * Converts the keys to the headers required for CSV output.
        * Unpacks the ``Input`` in `job_record['input']` so that there is one dict entry
          for each input coordinate.
        * Formats the ``JobStatus`` for CSV output.
        """

        # Rename keys
        restructured_record = {
            new_key: job_record[old_key]
            for old_key, new_key in self._HEADER_MAPPER.items()
        }

        # Unpack input coordinates
        input_coords = {
            self._make_input_coord_header(i): x
            for i, x in enumerate(restructured_record[self._INPUT_HEADER])
        }
        restructured_record |= input_coords
        del restructured_record[self._INPUT_HEADER]

        # Format status
        restructured_record[self._STATUS_HEADER] = format_status(
            restructured_record[self._STATUS_HEADER]
        )

        return restructured_record

    def _make_input_coord_header(self, idx: int) -> str:
        """Make a heading for an input coordinate based on the index of the coordinate."""

        return f"{self._INPUT_HEADER}_{idx + 1}"

    def _write_to_csv(self, jobs: list[dict[str, Any]], csv_file: TextIOWrapper) -> None:
        """Write details of jobs to an open CSV file."""

        field_names = (
            [self._JOBID_HEADER]
            + [self._make_input_coord_header(i) for i in range(self._app.input_dim)]
            + [self._STATUS_HEADER, self._RESULT_HEADER]
        )
        writer = csv.DictWriter(csv_file, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(jobs)

    @cmd2.with_argparser(add_interface_parser)
    def do_add_interface(self, args) -> None:
        """Add a hardware interface to the workspace."""

        try:
            interface_settings = json.load(args.file)

            # Check if interface is already in workspace
            interface_name = interface_settings["name"]
        except json.JSONDecodeError as e:
            self._render_error(f"Error reading interface settings: {e}")
            return None

        # Ask for interface type
        #ToDo: refactor to separate method - pick hardware interface
        while True:
            self.poutput("Select the hardware interface type of this interface:")
            for option, (display_name, _) in Cli.INTERFACE_FACTORIES.items():
                self.poutput(f"  {option}: {display_name}")

            factory_choice = input("Enter the number corresponding to your choice: ")
            selected_factory = Cli.INTERFACE_FACTORIES.get(factory_choice)

            if not selected_factory:
                self.poutput("Invalid choice, please try again.")
                continue

            else:
                break

        display_name, factory_cls = selected_factory
        factory = factory_cls()

        # load hardware interface
        self.poutput("Setting up hardware...")
        factory.load_hardware_parameters(args.file.name)
        hardware_interface = factory.create_hardware()


        hardware_params_filename = (
                self._hardware_params_prefix + hardware_interface.name + ".json"
        )
        hardware_params_file = self._workspace_dir / hardware_params_filename
        factory.serialise_hardware_parameters(hardware_params_file)

        general_settings_file = self._workspace_dir / "settings.json"

        # Read in general settings from file
        general_settings = read_settings_json(general_settings_file)

        # Get interface details from settings
        interface_details = general_settings["interfaces"]

        # Add new interface to interface details
        interface_details[interface_name] = {
            "factory": factory_cls.__name__,
            "params": hardware_params_filename,
        }

        # Write updated settings to file
        write_settings_json(
            {
                "interfaces": interface_details,
                "input_dim": general_settings["input_dim"],
            },
            general_settings_file,
        )

        self.poutput(f"Thanks -- new hardware interface '{interface_name}' added to workspace '{self._workspace_dir}'.")

        self.poutput("Adding interface to job manager...")
        self._app.add_interface(hardware_interface)
        self.poutput("Interface added.")



def clean_input_string(string: str) -> str:
    """Remove leading and trailing whitespace and quotes from a string.

    Examples
    --------

    Remove leading/trailing whitespace:

    >>> clean_input_string("  foo\n")
    'foo'

    Remove leading/trailing quotes (single and double):

    >>> clean_input_string("\"foo'")
    'foo'

    """
    return string.strip().strip("'\"")


def parse_statuses_string_to_set(
    statuses: str, empty_to_all: bool = False
) -> set[JobStatus]:
    """Convert a string listing of job statuses to a set.

    Before converting, the input string is cleaned by removing any leading and
    trailing whitespace and any quotation marks.

    Parameters
    ----------
    statuses : str
        A comma-separated list of job statuses. The statuses should match the names of
        the corresponding enums, with spaces represented either as whitespace or
        underscores and with matching done case-insensitively. (See examples.)
    empty_to_all : bool, optional
        (Default: False) Whether to interpret the empty list as representing no job
        statuses (``False``) or all possible job statuses (``True``).

    Returns
    -------
    set[JobStatus]
        A set of the job statuses from the string.

    Examples
    --------

    Provide statuses as a comma-separated string. Note that the case doesn't matter
    and any leading/trailing whitespace is removed:

    >>> cli._parse_statuses_string_to_set("cancelled,   failed")
    {<JobStatus.CANCELLED: 'Cancelled'>, <JobStatus.FAILED: 'Failed'>}

    Spaces in the status can be represented as whitespace or underscores:

    >>> cli._parse_statuses_string_to_set("pending submit")
    {<JobStatus.PENDING_SUBMIT: 'Pending submit'>}
    >>> cli._parse_statuses_string_to_set("pending_submit")
    {<JobStatus.PENDING_SUBMIT: 'Pending submit'>}

    By default, providing an empty string returns the empty set, but setting
    ``empty_to_all = True`` will cause the full set of job statuses to be returned
    instead:

    >>> cli._parse_statuses_string_to_set("")
    set()
    >>> cli._parse_statuses_string_to_set("", empty_to_all=True) == set(JobStatus)
    True
    """
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


def parse_bool(result: str) -> Optional[bool]:
    """Convert a string to a boolean.

    Converts 'true' to ``True``, 'false' to ``False`` and any other string to
    ``None``.
    """
    if result == "true":
        return True
    elif result == "false":
        return False
    else:
        return None


def parse_inputs(inputs: Union[Sequence[str], TextIOWrapper]) -> list[tuple[float, ...]]:
    """Convert string representations of simulator inputs to tuples of floats.

    Any leading/trailing whitespace or quotes are removed from the string before parsing
    and strings that are empty after this cleaning are ignored.

    Parameters
    ----------
    inputs : Union[Sequence[str], TextIOWrapper]
        A sequence of strings that define simulator inputs as a comma-separated list of
        floats. These can be provided by an open text file.

    Returns
    -------
    list[tuple[float, ...]]
        The simulator inputs, as tuples of floats.

    Raises
    ------
    ParsingError
        If the between-comma components of any of the strings cannot be parsed as a float.

    Examples
    --------

    Parse simulation inputs from a list of strings:

    >>> parse_inputs(["1,2,3", "-1,-0.09,0.654"])
    [(1.0, 2.0, 3.0), (-1.0, -0.09, 0.654)]

    Leading/trailing whitespace and quotes (single and double) are removed before parsing:

    >>> parse_inputs(["  1, 2, 3\n"])
    [(1.0, 2.0, 3.0)]

    >>> parse_inputs(["'1,2,3'"])
    [(1.0, 2.0, 3.0)]

    Any strings containing only whitespace or quotes are ignored in the output:

    >>> parse_inputs(["", "\"\""])
    []
    """

    if inputs:
        try:
            cleaned_inputs = map(clean_input_string, inputs)
            return [
                tuple(map(float, x.split(","))) for x in cleaned_inputs if not x == ""
            ]
        except ValueError as e:
            raise ParsingError(e)
    else:
        return []


def parse_job_ids(job_ids: Sequence[str]) -> tuple[JobId, ...]:
    """Parse a sequence of string job IDs to a tuple of ``JobId``s.

    Removes any repeated IDs and orders the returned IDs by string ordering.

    Parameters
    ----------
    job_ids : Sequence[str]
        A sequence of job IDs, as strings.

    Returns
    -------
    tuple[JobId, ...]
        The unique job IDs as ``JobId`` objects.

    Raises
    ------
    ParsingError
        If one of the supplied IDs does not define a valid job ID.
    """
    parsed_ids = set()
    for id_ in job_ids:
        try:
            parsed_ids.add(JobId(id_))
        except ValueError:
            raise ParsingError(
                f"{id_} does not define a valid job ID: should be a non-negative integer."
            )

    return tuple(sorted(parsed_ids, key=str))


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
