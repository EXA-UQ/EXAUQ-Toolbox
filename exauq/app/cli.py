import sys

import cmd2


class App(cmd2.Cmd):
    """The exauq command line application for managing jobs."""

    def __init__(self):
        super().__init__()
        self.prompt = "(exauq)> "


def main():
    app = App()
    sys.exit(app.cmdloop())


if __name__ == "__main__":
    main()
