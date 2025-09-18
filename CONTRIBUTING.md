# Contribution Guidelines

We use [uv](https://docs.astral.sh/uv/) for dependency management during development. UV is easily installed __without__
the need for a virtual environment or `conda` environment, but one can install it inside a `conda` environment too.

## Development model

We use a fork-and-merge development model (see an introduction
[here](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks) if needed).

The steps include:

1.  Creating a personal fork of `ml4gw` repository in your personal namespace.
2.  Creating a branch in your fork and make changes on it.
3.  Opening a pull request.

## Installation and proposing patches
Once `uv` is installed, clone your _fork_ using
```bash
git clone https://github.com/<namespace>/ml4gw
```
Track the upstream using
```bash
git remote add upstream https://github.com/ML4GW/ml4gw.git
```
Create a branch for your work
```bash
git checkout -b my-new-branch
```
Install the project using
```bash
uv sync --all-extras
```
If there is no virtual environment detected, `uv` will create a `.venv` directly at
the root of the repository and install all dependencies there. One can activate this
virtual environment using `source .venv/bin/activate`. However, for most cases, one
can use the `uv run` entrypoint. For example,
```bash
uv run pytest -v
```
will run the unittests for the project.

Once you made the changes and added the relevant tests, commit the changes and push
to the branch to your fork and create a pull request. You can check the patch
compared to `main` using
```bash
git fetch upstream main
git diff upstream/main
```
You have to be on your working branch for the diff to show correctly.

Also note that there is a code formatting check as a part of the continuous integration.
This can be done by installing the [pre-commit hook](https://pre-commit.com/)
which is a part of the development dependencies, and should run with a `git commit`.
However, it can be installed and run manually as
```bash
uv run pre-commit install
uv run pre-commit run --all-files
```
which will apply the necessary formatting, or indicate the issue for you to fix
if an automatic fix is not possible.

## Good practices

1. Please ensure that code patches are modular and preferably small. If the proposed
   change involves more than one feature, break it up into multiple pull requests.
2. When adding a new feature, for example a new waveform or a new pre-processing
   routine, add unittests that demonstrate the API and dimensions of inputs and outputs.
   If relevant, add tests that demonstrate scientific correctness, for example, a
   limiting behavior.
3. It is possible (and likely) that the librarians will ask for additional tests to cover 
   a new feature beyond what is covered in the unittests. Note that every feature also 
   adds maintenance costs, hence the usecase of a new feature will also need to well 
   motivated for gravitational-wave data analysis. For example, if you have used your 
   personal fork of `ml4gw` and performed an analysis that is published, it can be a strong
   usecase.
4. Please [cite](/CITATION.cff) the repository if you have used `ml4gw` in your work.

## Documentation
The documentation page is rendered using [sphinx-autodoc](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html). Please add
[google style docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
to new functions, classes, other routines that you add. If you add a new feature,
ensure that the documentation shows an example usage, for example, the dimensions
of inputs and outputs, or the expected result of a class method.

One can build the documentation locally by installing the `docs` dependency group,
```bash
uv sync --group docs
```
and then run the following at the root of the repo to create the html pages under
`docs/_build`:
```bash
uv run sphinx-build ./docs .docs/_build
```
Open the `index.html` in a browser and navigate to the section of added documentation
to check the rendering.
