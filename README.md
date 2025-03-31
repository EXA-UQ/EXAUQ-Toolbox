# EXAscale Uncertainty Quantification Toolbox (EXAUQ-Toolbox)

[![Tests](https://img.shields.io/github/actions/workflow/status/EXA-UQ/EXAUQ-Toolbox/test-package.yml?label=tests&logo=github)](https://github.com/EXA-UQ/EXAUQ-Toolbox/actions)
[![Python](https://img.shields.io/badge/Python-3.10%20|%203.11%20|%203.12%20|%203.13-blue?logo=python)](https://www.python.org/)
[![Docs](https://img.shields.io/badge/docs-online-blue)](https://exa-uq.github.io/EXAUQ-Toolbox/)
[![License](https://img.shields.io/github/license/EXA-UQ/EXAUQ-Toolbox)](https://github.com/EXA-UQ/EXAUQ-Toolbox/blob/main/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Import sorting: isort](https://img.shields.io/badge/imports-isort-ef8336.svg)](https://pycqa.github.io/isort/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://pre-commit.com/)
![Last commit](https://img.shields.io/github/last-commit/EXA-UQ/EXAUQ-Toolbox)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15005642.svg)](https://doi.org/10.5281/zenodo.15005642)


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
as part of project ExCALIBUR, grant number EP/W007886/1.

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

## Citation

If you use the EXAUQ-Toolbox in your work, please cite the appropriate version via Zenodo:

> Hawes, T., Johns, M., White, H., Salter, J., Olivier, E., Kimpton, L., Xiong, X., & Challenor, P. (2025).  
> *EXAUQ-Toolbox* [Computer software]. https://doi.org/10.5281/zenodo.15005642

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15005642.svg)](https://doi.org/10.5281/zenodo.15005642)
  
## Installing the latest release version

We strongly recommend using virtual environments (e.g. Conda environments) to
manage packages when using the EXAUQ-Toolbox in your work.

To install the `exauq` package from the toolbox, run the following command (after
activating the virtual environment, if necessary):

``` bash
python -m pip install "exauq @ git+https://github.com/EXA-UQ/EXAUQ-Toolbox.git"
```
This will install the latest version of `exauq` currently on the Git repository.
If you already have `exauq` installed and wish to update it to the latest version, run

``` bash
python -m pip install --force-reinstall "exauq @ git+https://github.com/EXA-UQ/EXAUQ-Toolbox.git"
```

If you wish to use a specific version of `exauq`, then simply alter the version at the end of the command. 

```bash
python -m pip install "exauq @ git+https://github.com/EXA-UQ/EXAUQ-Toolbox.git@v0.1.0"
```

## Viewing documentation

Documentation for the EXAUQ-Toolbox is available online at 
[https://exa-uq.github.io/EXAUQ-Toolbox/](https://exa-uq.github.io/EXAUQ-Toolbox/). 
To view it, you can run the following command within a terminal (after activating the environment 
where you installed the toolbox, if relevant):

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
poetry install --with=dev
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
**Note:** As of [Poetry 2.0](https://python-poetry.org/blog/announcing-poetry-2.0.0/#poetry-export-and-poetry-shell-only-available-via-plugins),
the `poetry shell` command has been removed. If you are using Poetry 2.0 or newer, please refer to the [official documentation](https://python-poetry.org/docs/managing-environments/#bash-csh-zsh)
for guidance on managing virtual environments.

Finally, to use the automatic pre-commit hooks designed for linting, run:

```bash
pre-commit install
```

this will be required to pass the linting checks for PRs. 

### Updating dependencies

To update Python package dependencies to the latest versions that are consistent
with the versioning constraints specified in `pyproject.toml`, run

```bash
poetry update
```

This will update the `poetry.lock` file.
