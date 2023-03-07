# EXAscale Uncertainty Quantification Toolbox (EXAUQ-Toolbox)


## Installation

## Installation for development

EXAUQ-Toolbox is written in Python and uses [Poetry](https://python-poetry.org/)
for package development and dependency management.

Assuming you have Python and Poetry installed, install Python package
dependencies by running the following from the root project folder:

```bash
$ poetry install
```

This will install packages into a dedicated virtual environment, according
to the versions specified in the `poetry.lock` file. The `exauq`
package is installed in [editable mode](https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs).
To run Python scripts / tooling within this virtual environment, either use 

```bash
$ poetry run <command-to-run>  # e.g. poetry run python foo.py
                               # e.g. poetry run black
```

or [activate the virtual environment](https://python-poetry.org/docs/basic-usage/#activating-the-virtual-environment)
e.g. by creating a nested shell:

```bash
$ poetry shell
```

### Updating dependencies

To update Python package dependencies to the latest versions that are consistent
with the versioning constraints specified in `pyproject.toml`, run

```bash
$ poetry update
```

## Software architecture

The following diagram gives an indication of the main classes used within the
software.

![architecture](resources/images/architecture.png)
