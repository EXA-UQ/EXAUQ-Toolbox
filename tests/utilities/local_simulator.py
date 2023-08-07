import pathlib
import shutil

from exauq.core.hardware import HardwareInterface


def main():
    workspace_dir = pathlib.Path("./sim123")
    try:
        _make_workspace(workspace_dir)
        _watch()
    except KeyboardInterrupt:
        pass
    finally:
        print(f"Cleaning up directory {workspace_dir.absolute()} ...")
        _clean_up_workspace(workspace_dir)
        print("Bye!")


def _make_workspace(workspace_dir: pathlib.Path):
    workspace_dir.mkdir(exist_ok=True)
    (workspace_dir / "inputs").mkdir(exist_ok=True)
    (workspace_dir / "outputs").mkdir(exist_ok=True)


def _watch():
    print("Simulator running, use Ctrl+C to kill.")
    while True:
        pass


def _clean_up_workspace(workspace_dir: pathlib.Path):
    if workspace_dir.exists():
        shutil.rmtree(workspace_dir)


class LocalHardware(HardwareInterface):
    def __init__(self):
        pass

    def submit_job(self, job):
        pass

    def get_job_status(self, job_id):
        pass

    def get_job_output(self, job_id):
        pass

    def cancel_job(self, job_id):
        pass

    def wait_for_job(self, job_id):
        pass


if __name__ == "__main__":
    main()
