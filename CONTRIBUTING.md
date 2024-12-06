# EXAUQ-Toolbox Contributing Guide

Thank you for considering contributing to the EXAUQ-Toolbox, a warm welcome!
Please read this contributing guide in full (and refer back to it when necessary) before
opening issues/creating PRs. This will help us review issues / answer your questions more
efficiently and hence help you have quicker responses!

To download for a developer please do check out the [README](README.md) instructions and ensure you're running 
`Python >= 3.10`. 

## Workflow

Within the EXAUQ-Toolbox we champion reproducible research, which as part of that, good documentation of changes and
test driven development lie at the heart of our ideals. Whilst this may seem a little tedious at times, this
ensures the integrity and credibility of the toolbox which, currently, has a very small team so please do follow along!

If you wish to contribute but are not used to (or want a refresher) to git version control, we highly recommend following 
the [introduction to version control](https://coding-for-reproducible-research.github.io/CfRR_Courses/individual_modules/section_landing_pages/introduction_to_version_control.html)
guide which gives good clear explanations for how to use GitHub most effectively and with best practices. Take particular note of the section for [keeping your email private](https://coding-for-reproducible-research.github.io/CfRR_Courses/individual_modules/introduction_to_version_control/configuring_git.html#keeping-your-email-private) when setting up GitHub as this is a public repository. 

### Issues

The key rule here is **1 question/bug/feature = 1 issue**.

Please feel free to raise multiple issues (within reason), 
but stick to one singular thing in an issue. It should be structured as follows:

1) **Motivation:** Why am I raising this as an issue? It could be a question, bug or feature request, but please let us know why you
are raising it and how it will help either the toolbox or a user of the toolbox.
2) **Example:** How can I show this? In the case of a bug or feature request, please give a basic example that we can either: reproduce, in the case of a bug; use
as the basis for a test case if it is for a feature request.
3) **Acceptance Criteria:** How will we know that this issue is now closed? You could do this via bullet points/tasks or simply an explanation that gives
clear criteria as what your expectations are at implementation.

**Note:** How do I? questions are also valid issues but please do refer to the documentation first and if it is missing then we will likely add
this into the documentation somewhere or ensure it is clearer. 
 
**Discussions:** It may be that one of the maintenance team wish to clarify/discuss your issue further, please do keep an eye once the issue is
open to answer any questions promptly. 

Poorly written issues will likely not be resolved without further clarification and if none received
will likely be removed. However, this process takes time so please do ensure your issues conform to our requests. 

As it stands there is only a very small team maintaining the toolbox, therefore please do give some time for the issue to be resolved. If
you wish to follow up on your own issue please feel free to create a branch and pull request (PR) (see below). 

### Labels (Contributors Only)

If you are an external contributor, one of the maintainers will add labels to your issue upon reading it. 

When creating issues please feel free to add appropriate labels to the issue from the list provided. This helps the team quickly 
identify bugs from feature requests etc. without having to open issues. This may seem trivial but if there are a sudden influx of issues
it does help us sort through them more quickly for those that are urgent or assigning them to the right person. 

There is also a **"Good First Issue"** label. If you are new to the EXAUQ-Toolbox and wish to contribute, please
do take a look at these first as our team realise these are good issues to get you up to speed with contributing to the toolbox more easily. 

### Branches/Forks and Pull Requests 

**Note:** These instructions differ slightly depending on if you are a contributor within the repository or if you are externally wishing to contribute due to the necessity of write permissions to protect branches. However, there are very few differences!

As an **external contributor:**

If you wish to resolve an issue you will have to create a fork of the repository to your own GitHub account and then clone this repository so that you have write permissions to your own forked copy. Rather than using the main branch of this fork, create a new branch which (for minor fixes) fits the template **"iss{issue_number}-{issue_title}"**. Depending on the changes made and if they are fairly complex we suggest pulling this down locally. When you have tested and are happy with your work / current progress, you can open a pull request to merge the fork back into the original EXAUQ-Toolbox repository. **Ensure that your are attempting to merge into the `dev` branch and not the default `main`!** If you forget, it will be rebased and this could cause issues with your merge - although, these can of course be fixed.   

As an **internal contributor:**

If you wish to resolve an issue please create a new branch and label the branch **"iss{issue_number}-{issue_title}"**. We suggest opening 
the branches locally and then pushing through to the remote branch when you wish. **NOTE:** This does not have to be when you are 
finished and ready to submit your PR. 

**For Both:**
Draft/WIP PRs are fully encouraged so that the maintainers of the 
toolbox can see what you are working on, please just label the title of your PR **"[WiP]{Name_of_branch/fork}"**. 

**Most importantly: Your new branch should base (or at least have the root base if you are branching off a branch) into `dev` and not `main`.** `dev` is the development branch, `main` is
for when we release and only an admin will be merging dev into main upon a new release. 

Every PR should close at least 1 issue. Please link the issue(s) at the top of your PR using the following: **"Closes #{number_of_issue}."** to 
attach the issue to the PR. There should then be a good description of how this issue has been closed by your PR (or how far you have 
got and where you are going if it is a draft). A well written issue will make this a lot easier! You should also assign yourself to that PR in order
to make everyone aware that is being worked on. If you realise you won't finish your PR and want to leave it in a WiP state then unassign yourself from it
and drop a note into the discussion. Someone else may well pick it up from where you left off. 

Please do not begin work on a branch someone else is working on without their prior consent, it will likely just cause merge issues and slow down progress. 
If you wish to help on an already open PR, please first enter the discussion and seek permission from whoever is assigned to that PR. If there is no one assigned 
then drop a comment in the discussion explaining how far you wish to take the PR and, upon approval of a maintainer, assign yourself to it and feel free to work on it.

When you think you have finished the PR, **before requesting a review**, please remember to:
1) **Documentation:** Double check you have updated all of the relevant documentation including API, user guides / tutorials (see below).
2) **Test:** Run all of the unittests.
3) **Lint**: There are github workflows such as linting checks and document rebuilding that will occur. See below for the standards we use and
how to use the pre-commit hook if necessary.


### Test Driven Development

Within every PR, we expect test coverage built in for any adaptations of functions or features. These should be automated unit tests which 
will fail before you write your new code and pass once you have implemented a bug fix/feature etc. In this toolbox these are created using python's unittest module and
can be run simply using `python -m unittest discover tests` from within the poetry shell. Within the code base there are already many unit tests
to give ideas for exception raises, edge cases and mocking etc. It should also prove fairly logical where to put tests as they should lie in a file called test_filename
and then be placed within the testcase for the class the method sits within. If it is a standalone function then it should have its own test class created in the appropriate location.  

**Please take the time to test your code!** 

### Reviewing

Finally, when your PR is ready, change the name of the PR to **"[ReadyforReview]{branch_name}"** and request a review from one of the team. Currently, 
we are still finalising how long reviews will take (and this will depend on the number of PRs). However, 2 weeks is probably a reasonable request currently. 
It is worth noting that the first thing our reviewers will do is run the tests and linting pre-commit hook. PRs which fail either of these checks will be sent 
straight back without further review until these pass. 

### Pre-commit and GitHub Actions

Pre-commit hooks are built into this Toolbox which include [iSort](https://pycqa.github.io/isort/) and [black](https://github.com/psf/black). Each time
you commit to a branch these hooks **should** be run automatically unless you are editing non-python files. Please do not skip them, this will only cause 
your PR to fail the check later on, and it is your responsibility, not the reviewer, to run the pre-commit locally. To do so, you can always run
```pre-commit run --all-files``` and granted the checks are all working correctly this should only affect code you have edited. 

We also have initial GitHub Actions in place which help our team maintain the code more easily - these include: 

- **iSort and black:** Before your code is merged into dev it will undergo checks for [iSort](https://pycqa.github.io/isort/) (for import sorting)
and linted using [black](https://github.com/psf/black) - more specifically `Black --line-length 90`. See pre-commit notes.
- **mkdocs build:** The docs will be rebuilt ensuring that your API documentation is the latest version, is fully up to date and builds with
no warnings or errors. On release these will also be pushed up to the relative GitHub pages. 
- **Test:** Unit tests will be run on the current determined versions of Python `3.10, 3.11, 3.12` and `3.13` to ensure compatibility.

### Branch Protection Rules

We have branch protection rules in place to protect the `main` and `dev` branches. These require all checks to pass alongside approved reviews from 
approved maintainers of the toolbox. Currently, there are no other protections in place, however, it is possible that some will be in place on specific branches 
from time to time in special circumstances of development. 

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
the documentation is a two-step process (run from the repository root directory,
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
server for the docs' website. This will watch for changes to the markdown source in `docs/`
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
writing documentation. Tutorials are designed for learning the basics, whereas user guides should be goal-oriented, 
consisting of directions that guide the reader through a problem or towards a result. 
