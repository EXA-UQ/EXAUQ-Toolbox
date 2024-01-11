from typing import Callable

from exauq.core.modelling import Input, SimulatorDomain


def maximise(func: Callable[[Input], float], domain: SimulatorDomain) -> Input:
    raise NotImplementedError
