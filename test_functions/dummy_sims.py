from exauq.core.simulator import Simulator
from exauq.utilities.BgHandler import BgHandler


class DummySimLvl0(Simulator):
    """
    Simple level 0 dummy simulator
    """

    def __init__(self):
        super().__init__()
        self.JOBHANDLER = BgHandler(host="localhost", user="")
        self.COMMAND = "sleep 1"


class DummySimLvl1(Simulator):
    """
    Simple level 1 dummy simulator
    """

    def __init__(self):
        super().__init__()
        self.JOBHANDLER = BgHandler(host="localhost", user="")
        self.COMMAND = "sleep 2"


class DummySimLvl2(Simulator):
    """
    Simple level 2 dummy simulator
    """

    def __init__(self):
        super().__init__()
        self.JOBHANDLER = BgHandler(host="localhost", user="")
        self.COMMAND = "sleep 5"


class DummySimLvl3(Simulator):
    """
    Simple level 3 dummy simulator
    """

    def __init__(self):
        super().__init__()
        self.JOBHANDLER = BgHandler(host="localhost", user="")
        self.COMMAND = "sleep 10"
