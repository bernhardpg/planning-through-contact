# Planning Through Contact

## Installation (Linux and MacOS)

This repo uses Poetry for dependency management. To setup this project, first
install [Poetry](https://python-poetry.org/docs/#installation) and, make sure
to have Python3.10 installed on your system.

Then, configure poetry to setup a virtual environment that uses Python 3.10:

```python
poetry env use python3.10
```

Next, install all the required dependencies to the virtual environment with the
following command:

```python
poetry install -vvv
```

(the `-vvv` flag adds verbose output).

## Using external Drake build

Externally built Drake must be added to the Python path:

```python
export PYTHONPATH=~/software/drake-build/install/lib/python3.11/site-packages:${PYTHONPATH}
```

See [the docs](https://drake.mit.edu/from_source.html) for how to build the
Python bindings.

## Add this library to your pythonpath

Add this module to pythonpath:

```python
export PYTHONPATH=$PYTHONPATH:.
```

## Some experiments

Elimination of linear equality constraints:

```python
python experiments/planar_pushing/one_contact_mode.py
```

## Running (deprecated) demos

```python
poetry run python deprecated/run_demos.py --demo box_push
```

```python
poetry run python deprecated/run_demos.py --demo box_pickup
```

Planning for flipping up polytopes of different shapes:

```python
poetry run python experiments/object-flipup/polytope_flipup.py --num_vertices 3
poetry run python experiments/object-flipup/polytope_flipup.py --num_vertices 4
poetry run python experiments/object-flipup/polytope_flipup.py --num_vertices 5

```

## Running pre-commit hooks

The repo is setup to do automatic linting and code checking on every commit
through the use of pre-commits. To run all the pre-commit hooks (which will
clean up all files in the repo), run the following command:

```python
poetry run pre-commit run --all-files
```

## Running a single experiment

Create a config file specifying the experiment in `config` and run it using the
following command:

```python
python3 scripts/planar_pushing/run_planar_pushing_experiment.py --config-name single_experiment
```

where `single_experiment` should be replaced with your config name.

## Running hardware simulations

Make sure to download `lcm` and build it to a desired directory:
[lcm](https://github.com/lcm-proj/lcm). Then, add `lcm` to the pythonpath:

```python
export PYTHONPATH="$PYTHONPATH:/Users/bernhardpg/software/lcm/build/python"
```

TODO: @bernhardpg complete this

## Additional packages

Make sure to have graphviz installed on your computer. On MacOS, run the following
command:

```python
brew install graphviz
```
