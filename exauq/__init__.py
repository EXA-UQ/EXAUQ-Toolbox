"""
EXAscale Uncertainty Quantification Toolbox (EXAUQ-Toolbox)
==========================================================

The `exauq` package provides a comprehensive suite of tools for developing Gaussian Process (GP)
emulators to model complex computer simulations. A core feature is its support for multi-level
GP emulation, enabling efficient surrogate modeling of simulation hierarchies with varying levels
of fidelity. The highest-fidelity simulations in these hierarchies may require exascale computing
resources, while lower-fidelity approximations can be executed on conventional HPC clusters or
departmental servers.

Beyond emulation, `exauq` facilitates the management of computational resources for large-scale
simulations. It abstracts job submission, monitoring, and retrieval across both local and remote
hardware environments, streamlining simulation workflows for uncertainty quantification tasks.

Key Features
============
- **Multi-level GP emulation**: Train hierarchical Gaussian Process models for multi-fidelity simulations.
- **Experimental design**: Generate effective sample distributions using Latin hypercube and
  leave-one-out (LOO) adaptive sampling methods.
- **Bounded hyperparameter control**: Extends `mogp_emulator` with hyperparameter bounding capabilities.
- **Simulation job management**: Submit, monitor, and retrieve simulation results across distributed
  computing environments, including SSH-based remote execution.
- **Flexible hardware interfaces**: Abstract interactions with local and remote computational resources.
- **Offline documentation**: Integrated documentation viewer for offline use.

Subpackages
---------------------------------------------------------------------------------------------------------
- [`core`][exauq.core]:
Implements the mathematical and statistical methods for experimental design, GP emulation, and numerical
tolerance checks.

- [`sim_management`][exauq.sim_management]:
Provides infrastructure for managing and orchestrating simulation jobs across various computing
environments.


References
---------------------------------------------------------------------------------------------------------
- `mogp_emulator`: <https://github.com/alan-turing-institute/mogp-emulator>
- LOO Sampling:
  - Mohammadi, H. et al. (2022) "Cross-Validation-based Adaptive Sampling for Gaussian Process Models".
    DOI: <https://doi.org/10.1137/21M1404260>
  - Kimpton, L. M. et al. (2023) "Cross-Validation Based Adaptive Sampling for Multi-Level Gaussian
    Process Models". arXiv: <https://arxiv.org/abs/2307.09095>

Developed by the Research Software Engineering (RSE) team at the University of Exeter, UK,
as part of the ExCALIBUR project (EPSRC grant number EP/W007886/1).


"""
