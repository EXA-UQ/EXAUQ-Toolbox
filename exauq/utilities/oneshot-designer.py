from numbers import Real
from typing import Optional

from scipy.stats.qmc import LatinHypercube

from exauq.core.modelling import Input, SimulatorDomain

def oneshot(
    domain: SimulatorDomain, design_num: int, seed: Optional[int] = None   
) -> tuple[Input, float]:
    
    if not isinstance(domain, SimulatorDomain):
        raise TypeError(
            f'Expected domain to be of type SimulatorDomain, but received {type(domain)}'
        )
    
    if not isinstance(design_num, int):
        raise TypeError(
            f'Expected design_num to be of type int, but received {type(design_num)}'
        )

    if seed is not None and not isinstance(seed, int):
        raise TypeError(
            f'Expected seed to be None or of type int, but received {type(seed)}'
        )
    
    # Use the dimension of the domain in defining the Latin hypercube sampler.
    # Seed used to make the sampling repeatable.
    sampler = LatinHypercube(domain.dim, seed)
    lhs_array = sampler.random(n=design_num)

    # Rescaled into domain
    lhs_inputs = [domain.scale(row) for row in lhs_array]

    return lhs_inputs

