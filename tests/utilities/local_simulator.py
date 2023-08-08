import math
import pathlib
import shutil
import signal
import sys
import time
from numbers import Real
from typing import Optional, Union

from exauq.core.hardware import HardwareInterface
from exauq.core.modelling import Input

WORKSPACE = pathlib.Path("./sim_workspace")


def simulate(x: Input):
    """A mock simulator. Computes the value of a function after a pause, to
    imitate an intensive computation."""
    time.sleep(5)
    return x[1] + (x[0] ** 2) + (x[1] ** 2) - math.sqrt(2)


def main():
    """Main entry point into the application.

    The application evaluates a simulator on inputs as these are received. The
    simulator is defined by a function at the top this script. The application
    continually watches for new input files (defined as those with extension
    '.in') created within a workspace folder, running the simulator on the
    contents of input files as they are received. Outputs for each simulation
    are written as '.out' files in the same workspace folder, with the same name
    as the corresponding input file.

    The application runs continually until it receives an interrupt from the
    user, upon which it deletes the workspace folder and its contents before
    exiting.
    """

    signal.signal(signal.SIGINT, _shutdown)
    print("*** Simulator running, use Ctrl+C to stop. ***")
    _make_workspace()
    _watch()


def _shutdown(sig_number, stack_frame):
    """Shutdown the application by deleting the workspace directory. This is
    expected to be used as a callback to a keyboard interruption issued by the
    user."""
    print(f"Cleaning up workspace directory '{WORKSPACE}'")
    _clean_up_workspace()
    sys.exit(0)


def _make_workspace():
    """Create the workspace folder on the file system."""
    WORKSPACE.mkdir(exist_ok=True)


def _watch():
    """Monitor the workspace folder for new simulation jobs and run the
    simulator on any new ones, outputting the results to the workspace
    directory."""
    while True:
        jobs = _get_new_jobs()
        for job in jobs:
            print(f"Running simulation {_get_id(job)}...")
            _write_output(simulate(_get_input(job)), _get_id(job))
            print("Done.")


def _get_new_jobs() -> list[tuple[Input, str]]:
    """Gather new jobs from the workspace folder, as represented by new '.in'
    files."""
    input_files = WORKSPACE.glob("*.in")
    output_ids = _get_filenames(".out")
    new_input_files = [p for p in input_files if _extract_filename(p) not in output_ids]
    return [(_read_input(p), _extract_filename(p)) for p in new_input_files]


def _get_input(job) -> Input:
    """Get the simulator `Input` from a job."""
    return job[0]


def _get_id(job) -> str:
    """Get the ID of a job."""
    return job[1]


def _read_input(input_path: pathlib.Path) -> Input:
    """Create a new simulator input from the contents of an input file."""
    file_contents = input_path.read_text().strip()
    return Input(*(map(float, file_contents.split(","))))


def _get_filenames(extension: str, workspace_dir: pathlib.Path = WORKSPACE) -> set[str]:
    """Return the names of all files in a workspace directory with the given
    extension. In this case, the name returned does not contain the file
    extension."""
    return set(map(_extract_filename, workspace_dir.glob(f"*{extension}")))


def _extract_filename(p: pathlib.Path):
    """Get the file name from a path (without the extension)."""
    return p.name.split(".")[0]


def _write_output(y: Real, _id: str):
    """Write a simulator output to an output file with the given ID in the
    workspace directory. Raises a ``FileExistsError`` if the output file already
    exists."""
    output_file = pathlib.Path(WORKSPACE / f"{_id}.out")
    if output_file.exists():
        raise FileExistsError(
            f"Cannot write output file {output_file}: it already exists."
        )
    output_file.write_text(str(y))


def _clean_up_workspace():
    """Delete the workspace directory and its contents, if it exists."""
    if WORKSPACE.exists():
        shutil.rmtree(WORKSPACE)


class LocalSimulatorInterface(HardwareInterface):
    def __init__(self, workspace_dir: Union[str, pathlib.Path]):
        self._workspace_dir = pathlib.Path(workspace_dir)

    def submit_job(self, job: Input) -> str:
        job_id = self._make_job_id()
        input_file = pathlib.Path(self._workspace_dir / f"{job_id}.in")
        content = ",".join(map(str, job))
        input_file.write_text(content)
        return job_id

    def _make_job_id(self):
        """Make a new ID for the job, based on the jobs that have already been
        submitted in the workspace directory."""
        previous_jobs = _get_filenames(".in", self._workspace_dir)
        last_job = max(previous_jobs) if previous_jobs else 0
        return str(int(last_job) + 1)

    def get_job_status(self, job_id: str) -> bool:
        pass

    def get_job_output(self, job_id: str) -> Optional[Real]:
        output_file = pathlib.Path(self._workspace_dir / f"{job_id}.out")
        if output_file.exists():
            return float(output_file.read_text())
        return None

    def cancel_job(self, job_id: str):
        pass

    def wait_for_job(self, job_id: str):
        pass


if __name__ == "__main__":
    main()
