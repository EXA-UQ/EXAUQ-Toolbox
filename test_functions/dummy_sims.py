from exauq.core.simulator import Simulator
from exauq.utilities.BgHandler import BgHandler


class DummySimLvl0(Simulator):
    """
    Simple level 0 dummy simulator
    """

    JOBHANDLER = BgHandler(host="localhost", user="")
    COMMAND = "sleep 1"


class DummySimLvl1(Simulator):
    """
    Simple level 1 dummy simulator
    """

    JOBHANDLER = BgHandler(host="localhost", user="")
    COMMAND = "sleep 2"


class DummySimLvl2(Simulator):
    """
    Simple level 2 dummy simulator
    """

    JOBHANDLER = BgHandler(host="localhost", user="")
    COMMAND = "sleep 5"


class DummySimLvl3(Simulator):
    """
    Simple level 3 dummy simulator
    """

    JOBHANDLER = BgHandler(host="localhost", user="")
    COMMAND = "sleep 10"
