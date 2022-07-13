from enum import Enum
from exauq.core.simulator import Simulator


class EventType(Enum):
    """
    Simulation status
    """

    SUBMIT_SIM = 0
    POLL_SIM = 1

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