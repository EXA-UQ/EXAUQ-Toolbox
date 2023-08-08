import argparse
import dataclasses
import math
import os
import pathlib
import signal
import sys
import time
from numbers import Real
from typing import Optional, Union

from exauq.core.hardware import HardwareInterface
from exauq.core.modelling import Input

WORKSPACE = pathlib.Path("./simulator_workspace")
"""Default workspace directory for the simulator to use. The application
continually watches for new input files created within the workspace directory
and writes output files to the same directory."""

SIM_SLEEP = 5
"""Default amount of time, in seconds, that the simulator should sleep for
before evaluating its function on the inputs."""

CLEANUP = False
"""Whether the simulator's workspace directory should be deleted upon application
shutdown."""


def simulate(x: Input, sim_sleep: float):
    """A mock simulator. Computes the value of a function after a pause, to
    imitate an intensive computation."""
    time.sleep(sim_sleep)
    return x[1] + (x[0] ** 2) + (x[1] ** 2) - math.sqrt(2)


def run_from_command_line():
    description = "A simple simulator to support experimentation with the EXAUQ-Toolbox."
    epilog = (
        "The application evaluates a simulator on inputs as these are received. "
        "The simulator is defined by a simple mathematical function, but "
        "but including a set amount of sleep to mimic more intensive work being "
        "done. Users submit and retrieve simulations by writing/reading files "
        "to/from a dedicated 'workspace' directory. The application continually "
        "watches this workspace for new input files (defined as those with "
        "extension '.in') created within, running the simulator on the "
        "contents of input files as they are received. Outputs for each "
        "simulation are written as '.out' files in the workspace, with the same "
        "name as the corresponding input file. The application runs continually "
        "until it receives an interrupt from the user. Optionally, the "
        "workspace folder and its contents will be deleted before exiting, so "
        "long as the directory didn't already exist before running the application."
    )
    parser = argparse.ArgumentParser(description=description, epilog=epilog)
    parser.add_argument(
        "--workspace",
        help=(
            "Path to a directory to act as the simulator's workspace. "
            f"Default: '{WORKSPACE.absolute()}'"
        ),
        default=WORKSPACE,
        nargs="?",
        const=WORKSPACE,
    )
    parser.add_argument(
        "--sim_sleep",
        help=(
            "The amount of time, in seconds, for the simulator to sleep before "
            f"evaluating its function. Default: {SIM_SLEEP}s."
        ),
        dest="sim_sleep",
        default=SIM_SLEEP,
        nargs="?",
        const=SIM_SLEEP,
        type=float,
    )
    parser.add_argument(
        "--cleanup",
        help=(
            "Whether to delete the workspace directory upon shutdown. Note that cleanup "
            "will only occur if the workspace directory doesn't exist before "
            f"running this application. Default: {CLEANUP}."
        ),
        dest="cleanup",
        default=CLEANUP,
        nargs="?",
        const=CLEANUP,
        type=bool,
    )
    args = parser.parse_args()
    run(args.workspace, args.sim_sleep, args.cleanup)


def run(workspace: str, sim_sleep: float, cleanup: bool):
    """Main entry point into the application.

    The application evaluates a simulator on inputs as these are received. The
    simulator is defined by a function at the top this script. The application
    continually watches for new input files (defined as those with extension
    '.in') created within a workspace folder, running the simulator on the
    contents of input files as they are received. Outputs for each simulation
    are written as '.out' files in the same workspace folder, with the same name
    as the corresponding input file.

    The application runs continually until it receives an interrupt from the
    user. Optionally, the application will delete the workspace folder and its
    contents before exiting. so long as the directory didn't already exist.
    """
    # Define shutdown handling
    _workspace = pathlib.Path(workspace)
    signal.signal(signal.SIGINT, _make_shutdown_handler(_workspace, cleanup))

    print("\n*** Simulator running, use Ctrl+C to stop. ***\n")
    if _workspace.exists():
        print(
            f"WARNING: workspace directory '{workspace}' already exists, "
            "contents may be overwritten.",
            file=sys.stderr,
        )

    _make_workspace(_workspace)
    _watch(_workspace, sim_sleep)


def _make_shutdown_handler(workspace: pathlib.Path, cleanup: bool):
    """Make a handler for a signal event (typically, upon an interrupt being
    issued by the user). If the workspace doesn't already exist and
    `cleanup=True` then the workspace will be deleted upon shutdown. Otherwise,
    exit with 0 exit code."""
    if cleanup and not workspace.exists():

        def _shutdown(sig_number, stack_frame):
            """Shutdown the application by deleting the workspace directory. This is
            expected to be used as a callback to a keyboard interruption issued by the
            user."""
            if workspace.exists():
                print(f"Cleaning up workspace directory '{workspace}'")
                _cleanup_workspace(workspace)
            sys.exit(0)

        return _shutdown

    return lambda x, y: sys.exit(0)


def _cleanup_workspace(workspace: pathlib.Path):
    """Delete the contents of a workspace."""
    contents = workspace.glob("*")
    try:
        for path in contents:
            os.remove(path)
        os.rmdir(workspace)
    except Exception as e:
        raise PermissionError(f"Could not clean up workspace: {str(e)}")


def _make_workspace(workspace: pathlib.Path):
    """Create a given workspace folder on the file system."""
    workspace.mkdir(exist_ok=True)


def _watch(workspace: pathlib.Path, sim_sleep: float):
    """Monitor the workspace folder for new simulation jobs and run the
    simulator on any new ones, outputting the results to the workspace
    directory. The simulator sleeps for the given number of seconds as part of
    its execution."""
    while True:
        jobs = _get_new_jobs(workspace)
        for job in jobs:
            print(f"Running simulation {job.id}...")
            _write_output(simulate(job.input, sim_sleep), job.id, workspace)
            print("Done.")


def _get_new_jobs(workspace: pathlib.Path) -> list["Job"]:
    """Gather new jobs from the workspace folder, as represented by new '.in'
    files."""
    input_files = workspace.glob("*.in")
    output_ids = _get_filenames(".out", workspace)
    new_input_files = [p for p in input_files if _extract_filename(p) not in output_ids]
    return [Job.from_file(p) for p in new_input_files]


def _write_output(y: Real, _id: str, workspace: pathlib.Path):
    """Write a simulator output to an output file with the given ID in the
    workspace directory. Raises a ``FileExistsError`` if the output file already
    exists."""
    output_file = pathlib.Path(workspace / f"{_id}.out")
    if output_file.exists():
        raise FileExistsError(
            f"Cannot write output file {output_file}: it already exists."
        )
    output_file.write_text(str(y))


def _get_filenames(extension: str, workspace: pathlib.Path) -> set[str]:
    """Return the names of all files in a workspace directory with the given
    extension. In this case, the name returned does not contain the file
    extension."""
    return set(map(_extract_filename, workspace.glob(f"*{extension}")))


def _extract_filename(p: pathlib.Path):
    """Get the file name from a path (without the extension)."""
    return p.name.split(".")[0]


@dataclasses.dataclass(frozen=True)
class Job(object):
    id: str
    input: Input

    @classmethod
    def from_file(cls, path: pathlib.Path):
        """Make a simulation job from a given input file."""
        file_contents = path.read_text().strip()
        _input = Input(*(map(float, file_contents.split(","))))
        return cls(id=_extract_filename(path), input=_input)


class LocalSimulatorInterface(HardwareInterface):
    """An interface to a local simulator.

    The local simulator is a command line application intended for testing
    purposes. It runs in its own process and the means of communicating with it
    (e.g. submitting jobs and retrieving output values) is by modifying the
    contents of the application's 'workspace' directory. For example, to submit
    a new simulation job, write the input to be evaluated to a '.in' file in
    the workspace, with filename being the ID of the job (e.g. '1.in'). When
    the simulation has completed, the output will be written to a corresponding
    output file in the workspace directory (e.g. to '1.out').

    Parameters
    ----------
    workspace_dir : str or pathlib.Path, optional
        A path to the directory of the local simulator's workspace (see above).
        Defaults to the value of the `WORKSPACE` constant defined in this
        module.
    """

    def __init__(self, workspace_dir: Union[str, pathlib.Path] = WORKSPACE):
        self._workspace_dir = pathlib.Path(workspace_dir)

    def submit_job(self, job: Input) -> str:
        """Submit a new job by writing it as an input file in the workspace
        directory."""
        job_id = self._make_job_id()
        input_file = pathlib.Path(self._workspace_dir / f"{job_id}.in")
        content = ",".join(map(str, job))
        input_file.write_text(content)
        return job_id

    def _make_job_id(self):
        """Make a new ID for the job, based on the jobs that have already been
        submitted in the workspace directory."""
        previous_job_ids = _get_filenames(".in", self._workspace_dir)
        last_job_id = max(previous_job_ids) if previous_job_ids else "0"
        return str(int(last_job_id) + 1)

    def get_job_output(self, job_id: str) -> Optional[Real]:
        """Get the output from the simulation with given job ID, if it exists,
        or return ``None`` otherwise."""
        if self.get_job_status(job_id) == 1:
            return float(self._output_path(job_id).read_text())
        return None

    def _output_path(self, job_id: str):
        return pathlib.Path(self._workspace_dir / f"{job_id}.out")

    def get_job_status(self, job_id: str) -> int:
        """Get the status of job, returning 1 if the job has been completed and
        0 otherwise."""
        if self._output_path(job_id).exists():
            return 1
        return 0

    def cancel_job(self, job_id: str):
        pass

    def wait_for_job(self, job_id: str):
        pass


if __name__ == "__main__":
    run_from_command_line()
