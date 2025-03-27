# EXAUQ-Toolbox: Uncertainty Quantification at the Exascale

Welcome to the documentation for the EXAUQ-Toolbox. The toolbox is a collection of
utilities that support the development of emulators of complex computer simulations. A key
feature is support for fitting multi-level Gaussian Process emulators for hierarchies of
simulation models. Here, simulations in the hierarchy are arranged according to differing
levels of model fidelity, with the simulation at the apex of the hierarchy potentially
requiring exascale computing to complete. For example, runs of the highest-fidelity
simulation may be run on an exascale computer, whereas a lower-fidelity, but cheaper,
simulation may be run on a more conventional HPC, departmental server or personal laptop.

As well as implementing statistical methods for training multi-level emulators, the
EXAUQ-Toolbox includes a command line application that takes care of the submission and
management of remote jobs for simulators.

<div class="grid cards" markdown>

-   :octicons-beaker-24:{ .lg .middle } **Experimental Design**

    ---
    Tutorials and user guides for using the EXAUQ-Toolbox for experimental design
    in your own analyses.

    [:octicons-arrow-right-24: Go to experimental design guides](./designers/index.md)

-   :simple-gnubash:{ .lg .middle } **Command Line Application**

    ---

    Manage simulation jobs from the comfort of your own laptop.

    [:octicons-arrow-right-24: Go to CLA reference](./cli/index.md)

-   :material-cogs:{ .lg .middle } **API Reference**

    ---
    Details of the objects, functions etc. provided by the exauq package.

    <!-- Following is a link to fake markdown doc but it will get converted to correct html page link -->
    [:octicons-arrow-right-24: Go to API reference](./api/__init__.md)

</div>