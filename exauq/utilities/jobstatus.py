from enum import Enum


class JobStatus(Enum):
    """
    Simulation status
    """

    INSTALLED = 0
    INSTALL_FAILED = 1
    SUBMITTED = 2
    SUBMIT_FAILED = 3
    IN_QUEUE = 4
    RUNNING = 5
    FAILED = 6
    SUCCESS = 7
