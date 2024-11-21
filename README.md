# EXAscale Uncertainty Quantification Toolbox (EXAUQ-Toolbox)

The EXAUQ-Toolbox is a collection of packages and utilities that supports the development of
emulators of complex computer simulations. A key feature is support for fitting
multi-level Gaussian Process emulators for hierarchies of simulation models. Here, simulations in the
hierarchy are arranged according to differing levels of model fidelity, with the
simulation at the apex of the hierarchy potentially requiring exascale
computing to complete. As well as implementing statistical methods for training multi-level
emulators, the EXAUQ-Toolbox takes care of managing the use of multiple computing
resources, which would typically be required when running simulations in the
multi-fidelity hierarchy. (For example, runs of the highest-fidelity simulation may
be run on an exascale computer, whereas a lower-fidelity, but cheaper, simulation may be
run on a more conventional HPC, or a departmental server).


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


## Developing documentation

### Tooling

The toolbox uses [MkDocs](https://www.mkdocs.org/) with the following plugins:
- `mkdocs-material`: [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
  theme.
- `mkdocstrings`: [Auto-generation of API docs](https://mkdocstrings.github.io/) from
  docstrings.
- `mkdocs-gen-files`: [Programmatically generate docs at build time](https://oprypin.github.io/mkdocs-gen-files/).
- `mkdocs-literate-nav`: [Specify navigation in markdown](https://github.com/oprypin/mkdocs-literate-nav),
  used to support API docs generation (following the mkdocstrings recipe,
  [Automatic code reference pages](https://mkdocstrings.github.io/recipes/#automatic-code-reference-pages)). 

### Building the docs

Currently, documentation is distributed within the toolbox for offline viewing. Building
the documentation is a two step process (run from the repository root directory,
either using `poetry run` or after first activating the Poetry environment):

1. Run `python scripts/build_notebooks.py`. This converts any Jupyter notebooks
   into markdown equivalents, which are ultimately what MkDocs will use to build the docs.
   By default, the notebooks are executed before converting to markdown, which provides a
   basic layer of testing for the documentation (i.e. it at least checks that the tutorial
   runs without crashing). To skip this, supply the `-n` option to
   `scripts/build_notebook.py`.
2. Run `mkdocs build` to build the documentation from the contents of the `docs`
   directory.

When developing documentation, you can instead use `mkdocs serve` to start a development
server for the docs website. This will watch for changes to the markdown source in `docs/`
and regenerate the web content automatically. A workflow which is more suitable for quick
feedback when writing docs:

1. Run `mkdocs serve` in a separate shell.
2. Edit markdown files in `docs/` and see the results get re-rendered live. If you edit
   Jupyter notebooks, then convert these to markdown using `scripts/build_notebooks.py` in
   order for them to be rendered by the development server. (Depending on what you're
   doing, you may want to use the `-n` option; see above.)


### Writing / editing docs

Documentation should be added to appropriate subdirectories of the `docs/` directory.
The layout is summarised as follows:

- `docs/index.md`: The root page of the documentation website.
- `docs/designers`: Tutorials and guides for doing experimental design.
- `docs/cli`: Tutorials and guides for using the `exauq` command line app for managing
  jobs.

Landing pages for sections are called `index.md` within the relevant directory. We also
separate out tutorials and user-guides (the former being more introductory and giving
a tour of the main features, while the latter go into more detail of specific topics,
while still having a 'how-to' focus).

Jupyter notebooks that need to be converted to markdown via the `scripts/build_notebooks.py`
script should go in `notebooks` subdirectories. For example,
`docs/designers/tutorials/notebooks` contains notebooks for tutorials on experimental
design. The `scripts/build_notebooks.py` script contains hardcoded filepaths specifying
where to find the notebooks and where to write the markdown versions.
 
