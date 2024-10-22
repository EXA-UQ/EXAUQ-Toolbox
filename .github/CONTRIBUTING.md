# EXAUQ-Toolbox Contributing Guide

Thank you for considering contributing to the EXAUQ-Toolbox, a warm welcome!
Please read this contributing guide in full (and refer back to it when necessary) before
opening issues/creating PRs. This will help us review issues / answer your questions more
efficiently and hence help you have quicker responses!

To download for a developer please do check out the [README](README.md) instructions and ensure you're running 
`Python >= 3.11`. 

## Workflow

Within the EXAUQ-Toolbox we champion reproducible research, which as part of that, good documentation of changes and
test driven development lie at the heart of our ideals. Whilst this may seem a little tedious at times, this
ensures the integrity and credibility of the toolbox which, currently, has a very small team so please do follow along!

### Issues

The key rule here is **1 question/bug/feature = 1 issue**. Please feel free to raise multiple issues (within reason), 
but stick to one singular thing in an issue. It should be structured as follows:

1) **Motivation:** Why am I raising this as an issue? It could be a question, bug or feature request, but please let us know why you
are raising it and how it well help either the toolbox or a user of the toolbox.
2) **Example:** How can I show this? In the case of a bug or feature request, please give a basic example that we can either, reproduce in the case of a bug or use
as the basis for a test case if it is feature based. 
3) **Acceptance Criteria:** How will we know that this issue is now closed? You could do this via bullet points/tasks or simply an explanation that gives
clear criteria as what your expectations are at implementation.

 
**Note discussion:** It may be that one of the maintenance team wish to clarify/discuss your issue further, please do keep an eye once the issue is
open to answer any questions promptly. 

Poorly written issues will likely not be resolved without further clarification and if none received
will likely be removed. However, this process takes time so please do ensure your issues conform to our requests. 

As it stands there is only 1 person maintaining the toolbox, therefore please do give some time for the issue to be resolved. If
you wish to follow up on your own issue please feel free to open a branch and then PR (see below). 

### Labels

When creating issues please feel free to add appropriate labels to the issue from the list provided. This helps the team quickly 
identify bugs from feature requests etc. without having to open issues. This may seem trivial but if there are a sudden influx of issues
it does help us sort through them more quickly for those that are urgent or assigning them to the right person. 

There is also a **"Good First Issue"** label. If you are new to the EXAUQ-Toolbox and wish to contribute, please
do take a look at these first as our team realise these are good issues to get you up to speed with contributing to the toolbox more easily. 

### Branches and Pull Requests

If you wish to resolve an issue please create a new branch and label the branch **"iss{number}-{issue_title}"**. We suggest opening 
the branches locally and then pushing through to the remote branch when you wish. **NOTE:** This does not have to be when you are 
finished and ready to submit your PR. Draft/WIP PRs are fully encouraged so that the maintaners of the 
toolbox can see what you are working on, please just label the title of your PR **"[WIP]{Name_of_branch}"**. 

Every PR should close at least 1 issue. Please link the issue(s) at the top of your PR using the following: **"Closes #issueNo."** to 
attach the issue to the PR. There should then be a good description of how this issue has been closed by your PR (or how far you have 
got and where you are going if it is a draft!). A well written issue will make this a lot easier!

When you think you have finished the PR please, **before requesting a review**, remember to:
1) **Documentation:** Double check you have updated all of the relevent documentation including API, user guides / tutorials (see below).
2) **Test:** Run all of the unittests.
3) There are (will be) github workflows such as linting and documents rebuilding that will occur before merging. (However, a quick
test of this yourself is also appreciated!). See below for the standards we use.

### Test Driven Development

Within every PR, we expect test coverage built in for any adaptations of functions or features. These should be automated unit tests which 
will fail before you write your new code and pass once you have implemented a bug fix/feature etc. These are created using python's unittest module and
can be run simply using `python -m unittest discover tests` from within the poetry shell. The best examples can be found in the codebase already 
to give ideas for exception raises, edge cases and mocking etc. 

**Please take the time to test your code!**

### Reviewing

Finally when your PR is ready, change the name of the PR to **"[ReadyforReview]{branch_name}"** and request a review from one of the team. Currently, 
we are still finalising how long reviews will take (and this will depend on the number of PRs). However, 2 weeks is probably a reasonable request currently. 

### Github Workflows

We have set up some initial workflows which help our team maintain the code more easily - these include: 

- **iSort and black:** Before your code is merged into dev it will undergo [iSort](https://pycqa.github.io/isort/) (for import sorting)
and linted using [black](https://github.com/psf/black) - more specifically `Black --line-length 90`.
- **mkdocs build:** The docs will be rebuilt ensuring that your API documentation is the latest version, is fully up to date and builds with
no warnings or errors.
- **Test:** Unit tests will be run on the current determined versions of Python `3.11, 3.12` and `3.13` to ensure compatibility. 

## Developing Documentation

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

### Building the Docs

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


### Writing / Editing Docs

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

### API

Given the usage of an auto-generated API docs, there are certain issues with formatting and rendering that 
mean we do not follow one particular style in its entirety, however we follow the [Numpydoc conventions
for docstrings](https://numpydoc.readthedocs.io/en/latest/format.html#style-guide) as closely as possible.
See also the section on [multi-line docstrings from PEP 257](https://peps.python.org/pep-0257/#multi-line-docstrings) for 
useful guidance on what to include. Note that this kind of documentation can be quite terse and is for reference only,
as we use guides and tutorials to show examples of objects in context following the [Diátaxis](https://diataxis.fr/) approach. 

The most common deviations are: 

- Remove incoming argument types from docstrings and rely on in-code type hints
- API reference links should **only** be placed into the "see also" section and not in the docstring body.

If you do include examples as part of the docstring then these should follow the [doctest](https://docs.python.org/3.10/library/doctest.html) format. 
In general, **there should be no errors or warnings through [doctest](https://docs.python.org/3.10/library/doctest.html) or building the docs with `mkdocs`**.  

We ask that before you request a review for your PR, you look through the online API documentation to check that any of your changes
are rendered correctly and any links/references etc. all work as intended. 

### User Guides & Tutorials

Good documentation consists of more than just API documentation. We already have a few tutorials 
introducing newcomers on how to use the toolbox to train emulators and do basic adaptive sampling 
(both single and multi-level). These tutorials only cover the most basic use cases, however. 
If you wish to contribute to the guides and tutorials, we follow the [Diátaxis](https://diataxis.fr/) approach to 
writing documentation. Tutorials are designed for learning the basics, where as user guides should be goal-oriented, 
consisting of directions that guide the reader through a problem or towards a result. 
