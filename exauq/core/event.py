from enum import Enum
from exauq.core.simulator import Simulator


class EventType(Enum):
    """
    Event types
    """

    INSTALL_SIM = 0
    RUN_SIM = 1
    POLL_SIM = 2


class Event:
    """
    Basic event class
    """

    def __init__(self, event_type: EventType, sim: Simulator) -> None:
        """
        Instantiate a basic event associated with a particular sim
        """
        self.type = event_type
        self.sim = sim
