from enum import Enum

class JobStatus(Enum):
    """
    Simulation status
    """
    WAITING = 0
    SUBMIT_FAILED = 1
    IN_QUEUE = 2
    RUNNING = 3
    FAILED = 4
    SUCCESS = 5