import argparse
import sys
from collections.abc import Sequence
from io import TextIOWrapper
from typing import Union

import cmd2

from exauq.app.app import App


class ParsingError(Exception):
    """Raised when errors arise from parsing command line arguments."""

    def __init__(self, e: Union[str, Exception]):
        self._base_msg = str(e)

    def __str__(self):
        return f"Error parsing args: {self._base_msg}"


class ExecutionError(Exception):
    """Raised when errors arise from the running of the application."""


class CLI(cmd2.Cmd):
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

    def __init__(self):
        super().__init__()
        self.app = App()
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

    def _render_submissions(self, submissions: dict[str, tuple[float, ...]]) -> str:
        return f"** Rendering of submissions {submissions}. **"

    @cmd2.with_argparser(submit_parser)
    def do_submit(self, args) -> None:
        """Submit a job to the simulator."""

        try:
            inputs = self._parse_inputs(args.inputs) + self._parse_inputs(args.file)
            submissions = self.app.submit(inputs)
            self.poutput(self._render_submissions(submissions))
        except ParsingError as e:
            self.perror(str(e))

    def do_status(self, args) -> None:
        """Get the status of simulation jobs."""

        _ = self.app.status()
        self.poutput("** Rendering of statuses. **")

    def do_result(self, args) -> None:
        """Retrieve the result of simulation jobs"""

        _ = self.app.result()
        self.poutput("** Rendering of results. **")


def main():
    app = CLI()
    sys.exit(app.cmdloop())


if __name__ == "__main__":
    main()
