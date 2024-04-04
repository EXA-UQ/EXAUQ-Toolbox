import argparse
import sys
from collections.abc import Sequence
from numbers import Real

import cmd2


class App:
    def __init__(self):
        pass

    def submit(self, inputs: Sequence[Sequence[Real]]) -> str:
        return f"** Submitted {inputs}. **"

    def status(self) -> str:
        return "** Shows status of some jobs. **"

    def result(self) -> str:
        return "** Gets the result of jobs. **"


class Cmd2AppWrapper(cmd2.Cmd):
    """The exauq command line application for managing jobs."""

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

    @cmd2.with_argparser(submit_parser)
    def do_submit(self, args) -> None:
        """Submit a job to the simulator."""

        inputs = []
        try:
            if args.inputs:
                inputs.extend(tuple(map(float, x.split(","))) for x in args.inputs)
            if args.file:
                inputs.extend(
                    tuple(map(float, x.split(","))) for x in args.file.readlines()
                )
            self.poutput(self.app.submit(inputs))
        except ValueError as e:
            self.perror(f"Could not parse inputs: {e}")
        finally:
            if args.file:
                args.file.close()

    def do_status(self, args) -> None:
        """Get the status of simulation jobs."""

        self.poutput(self.app.status())

    def do_result(self, args) -> None:
        """Retrieve the result of simulation jobs"""

        self.poutput(self.app.result())


def main():
    app = Cmd2AppWrapper()
    sys.exit(app.cmdloop())


if __name__ == "__main__":
    main()
