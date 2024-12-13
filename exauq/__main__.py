import argparse
import os
import pathlib
import subprocess
import sys
import webbrowser
from importlib.metadata import version, PackageNotFoundError

import exauq
import exauq.app.cli


def get_version() -> str:
    """Retrieve the version of exauq currently installed."""

    try:
        return version("exauq")
    except PackageNotFoundError:
        return "Package not found."


def launch_docs() -> None:
    """Open the documentation for the EXAUQ-Toolbox from a local static site."""

    doc_index = pathlib.Path(exauq.__file__).absolute().parent / "docs" / "index.html"

    if "WSL_DISTRO_NAME" in os.environ:

        result = subprocess.run(
            ["wslpath", "-w", doc_index], capture_output=True, text=True
        )
        path = result.stdout.strip()
        url = "file://" + path
        subprocess.run(["wslview", url])
    else:
        webbrowser.open(f"file://{doc_index}", new=2)  # open in new tab


def main():
    """The entry point into the EXAUQ command line application."""

    try:
        parser = argparse.ArgumentParser(
            description="Submit and view the status of simulations.",
        )
        parser.add_argument(
            "workspace",
            type=pathlib.Path,
            nargs="?",  # 0 or 1
            default=".exauq-ws",
            help="path to a directory for storing hardware settings and simulation results (defaults to '%(default)s')",
        )
        parser.add_argument(
            "-d",
            "--docs",
            action="store_true",
            help="open a browser at the EXAUQ documentation and exit",
        )
        parser.add_argument(
            "-v",
            "--version", 
            action="version",
            version=f"exauq {get_version()}",
            help="show the current installed version of the EXAUQ-Toolbox and exit",
        )

        args = parser.parse_args()

        if args.docs:
            sys.exit(launch_docs())
        else:
            cli = exauq.app.cli.Cli(args.workspace)
            sys.exit(cli.cmdloop())

    except KeyboardInterrupt:
        sys.exit(print())  # Use of print ensures next shell prompt starts on new line


if __name__ == "__main__":
    main()
