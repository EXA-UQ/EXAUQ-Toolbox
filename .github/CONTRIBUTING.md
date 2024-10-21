Thank you for considering contributing to the EXAUQ-TOOLBOX, a warm welcome!
Please read this contributing guide in full (and refer back to it when necessary) before
opening issues/creating PRs. This will help us review issues / answer your questions more
efficiently and hence help you have quicker responses!

## Issues



Types of issues (Main examples but not limited to):
1) Questions regarding usage of the toolbox etc. 
2) Bugs
3) Feature Requests

When writing an issue, particularly a bug/feature request, it should have some form 
of motivation, preferably an example (ie a bug is doing this but I expect this) or
I want a feature to do this in this case. Then it should have some form of acceptance criteria
ie. this issue can be closed if for example, bug is fixed, or feature implemented could solve
this and this for example. 

Poorly written issues will likely not be resolved without further clarification and if none received
will likely be removed. 

Please also link 1 bug/ 1 feature request to 1 issue. If you have multiple, that is great - but please
create multiple separate issues for these to be solved.

## Pull Requests

Fill in text about how to structure PRs

A PR should close at least 1 issue. Draft/WIP PRs are fully encouraged so that the maintaners of the 
toolbox can see what you are working on. Within the PR please clearly document which issue(s) you are closing
alongside a good description of how this issue has been closed by your PR. (Note: A well written issue will make this a lot
easier!)

When you think you have finished the PR please, **before requesting a review** remember to 
1) Double check you have updated all of the relevent documentation including API, user guides / tutorials
2) Run all of the unittests
3) There are (will be) github workflows such as linting and documents rebuilding that will occur before merging. (However, a quick
test of this yourself is also appreciated!). 

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
 

### User Guides & Tutorials

Fill in text here about how to write good user guides and tutorials following TH link advice (HW will find and write!)


## Labels

Fill in text about different labels that can be used within the Toolbox depending on what people want to contribute. 
(ie making use of the goodfirst issue label as a new contributor etc. )