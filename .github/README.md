# EXAscale Uncertainty Quantification Toolbox (EXAUQ-Toolbox)

The EXAUQ-Toolbox is a collection of packages and utilities that supports the development of
emulators of complex computer simulations. A key feature is support for fitting
multi-level Gaussian Process emulators for hierarchies of simulation models. Simulations in the
hierarchy are arranged according to differing levels of model fidelity, with the
simulation at the top of the hierarchy potentially requiring exascale
computing to complete. As well as implementing statistical methods for training multi-level
emulators, the EXAUQ-Toolbox takes care of managing the use of multiple computing
resources, which would typically be required when running simulations in the
multi-fidelity hierarchy. For example, runs of the highest-fidelity simulation may
be run on an exascale computer, whereas a lower-fidelity, but cheaper, simulation may be
run on a more conventional HPC, or a departmental server.

The EXAUQ-Toolbox welcomes contributors and users to raise issues/ideas. Please do take a look through 
our [support](SUPPORT.md) page to see how you can help and how we can help you in the most efficient way!

Software developed by the RSE team at the University of Exeter, UK and funded by the EPRSC (2021-2025)
as part of project ExCALIBUR. JS - ADD SPECIFIC GRANT NUMBER?

Many thanks to the code developers of the toolbox:

  - [Dr. Thomas Hawes](https://github.com/thawes-rse)
  - [Dr. Matt Johns](https://github.com/mbjohns)
  - [Mr. Harrison White](https://github.com/HarryWhiteRSE)
  - [Dr. Enrico Olivier](https://github.com/ricky-lv426)

Alongside the research team for the algorithms: 

  - [Prof. Peter Challenor](https://experts.exeter.ac.uk/22136-peter-challenor)
  - [Dr. James Salter](https://experts.exeter.ac.uk/26439-james-salter)
  - [Dr. Louise Kimpton](https://experts.exeter.ac.uk/28206-louise-kimpton)
  - [Dr. Xiaoyu Xiong](https://experts.exeter.ac.uk/27140-xiaoyu-xiong)
  
## Installing the latest development version

We strongly recommend using virtual environments (e.g. Conda environments) to
manage packages when using the EXAUQ-Toolbox in your work.

To install the `exauq` package from the toolbox, run the following command (after
activating the virtual environment, if necessary):

``` bash
python -m pip install "exauq @ git+https://github.com/EXA-UQ/EXAUQ-Toolbox.git@dev"
```
This will install the version of `exauq` currently on the Git repository's `dev` branch.

If you already have `exauq` installed and wish to update it, run

``` bash
python -m pip install --force-reinstall "exauq @ git+https://github.com/EXA-UQ/EXAUQ-Toolbox.git@dev"
```

## Viewing documentation

The EXAUQ-Toolbox ships with documentation for offline viewing. To view it, run
the following command within a terminal (after activating the environment where you
installed the toolbox, if relevant):

``` bash
exauq --docs
```

## Developing the package

### Installation

EXAUQ-Toolbox is written in Python and uses [Poetry](https://python-poetry.org/)
for package development and dependency management.

Assuming you have Python and Poetry installed, install Python package
dependencies by running the following from the root project folder:

```bash
poetry install
```

This will install packages into a dedicated virtual environment, according
to the versions specified in the `poetry.lock` file. The `exauq`
package is installed in [editable mode](https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs).
To run Python scripts / tooling within this virtual environment, either use 

```bash
poetry run <command-to-run>  # e.g. poetry run python foo.py
                             # e.g. poetry run black
```

or [activate the virtual environment](https://python-poetry.org/docs/basic-usage/#activating-the-virtual-environment)
e.g. by creating a nested shell:

```bash
poetry shell
```

### Updating dependencies

To update Python package dependencies to the latest versions that are consistent
with the versioning constraints specified in `pyproject.toml`, run

```bash
poetry update
```

This will update the `poetry.lock` file.
