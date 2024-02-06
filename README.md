# Planning Through Contact

## Installation (Linux and MacOS)

This repo uses Poetry for dependency management. To setup this project, first
install [Poetry](https://python-poetry.org/docs/#installation) and, make sure
to have Python3.11 installed on your system.

Then, configure poetry to setup a virtual environment that uses Python 3.11:

```python
poetry env use python3.11
```

Next, install all the required dependencies to the virtual environment with the
following command:

```python
poetry install -vvv
```

(the `-vvv` flag adds verbose output).

## Generating planar pushing plans

Currently, the main entrypoint for generating planar pushing plans is the
following script:

```python
python scripts/planar_pushing/create_plan.py
```

which takes a number of command line arguments. Add the flag `--help` for a
description of these.

## Running pre-commit hooks

The repo is setup to do automatic linting and code checking on every commit
through the use of pre-commits. To run all the pre-commit hooks (which will
clean up all files in the repo), run the following command:

```python
poetry run pre-commit run --all-files
```

## Running a single hardware experiment

Create a config file specifying the experiment in `config` and run it using the
following command:

```python
python3 scripts/planar_pushing/run_planar_pushing_experiment.py --config-name single_experiment
```

where `single_experiment` should be replaced with your config name.

## Running hardware simulations

Make sure to download `lcm` and build it to a desired directory:
[lcm](https://github.com/lcm-proj/lcm). Then, add `lcm` to the pythonpath, e.g. like this:

```python
export PYTHONPATH="$PYTHONPATH:/Users/bernhardpg/software/lcm/build/python"
```

TODO: @bernhardpg complete this

## (Optional) Additional packages

Make sure to have graphviz installed on your computer. On MacOS, run the following
command:

```python
brew install graphviz
```

## (Optional) Using external Drake build

Externally built Drake must be added to the Python path:

```python
export PYTHONPATH=~/software/drake-build/install/lib/python3.11/site-packages:${PYTHONPATH}
```

See [the docs](https://drake.mit.edu/from_source.html) for how to build the
Python bindings.
