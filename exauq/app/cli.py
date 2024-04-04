import sys

import cmd2


class App:
    def __init__(self):
        pass

    def submit(self) -> str:
        return "** Submitted something. **"

    def status(self) -> str:
        return "** Shows status of some jobs. **"

    def result(self) -> str:
        return "** Gets the result of jobs. **"


class Cmd2AppWrapper(cmd2.Cmd):
    """The exauq command line application for managing jobs."""

    def __init__(self):
        super().__init__()
        self.app = App()
        self.prompt = "(exauq)> "

    def do_submit(self, args) -> None:
        """Submit a job to the simulator."""

        self.poutput(self.app.submit())

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
