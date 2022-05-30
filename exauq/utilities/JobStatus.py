from enum import Enum

class JobStatus(Enum):
    """
    Simulation status
    """
    WAITING = 0
    SUBMITTED = 1
    SUBMIT_FAILED = 2
    IN_QUEUE = 3
    RUNNING = 4
    FAILED = 5
    SUCCESS = 6