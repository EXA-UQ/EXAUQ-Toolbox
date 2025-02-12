"""
exauq.core
===========

-------------------------------------------------------------------------------------------
The exauq core package contains everything required for the creation and training 
of both single level and multi-level deterministic Gaussian process (GP) emulators 
and leave-one-out (LOO) sampling methods on customisable designs. The GP emulation
is built upon the `mogp_emulator` while implementing the ability to bound hyperparameters. 
Spatial domains can be created and filled with effective sampled points alongside
repulsion points across the boundary. 

Tutorials on how to create a basic experimental design can be found within the 
Experimental Design tutorials. 

-------------------------------------------------------------------------------------------
Modules
=========

[`designers`][exauq.core.designers]:
    Create the experimental domain using either a simple oneshot Latin hypercube or through
    the LOO sampling methods for both single and multi-level GPs.

[`emulators`][exauq.core.emulators]:
    Provides the emulators to modify and train the GP and control hyperparameters.
    
[`modelling`][exauq.core.modelling]:
    Contains the objects required for utilising and training GP emulators alongside the
    LOO sampling methods for both single and multi-level GPs.

[`numerics`][exauq.core.numerics]:
    Numerical tolerance checks.


References 
-------------------------------------------------------------------------------------------

`mogp_emulator`:  <https://github.com/alan-turing-institute/mogp-emulator>

`LOO Sampling`: 

Mohammadi, H. et al. (2022) "Cross-Validation-based Adaptive
Sampling for Gaussian process models". DOI: <https://doi.org/10.1137/21M1404260>

Kimpton, L. M. et al. (2023) "Cross-Validation Based Adaptive Sampling for
        Multi-Level Gaussian Process Models". arXiv: <https://arxiv.org/abs/2307.09095>


"""
