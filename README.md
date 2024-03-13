# Planning Through Contact

**Note: This repo is still under active development and is updated continuously.**

---

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

#### (Mandatory) Installing customized Drake version

At the moment, this code relies on a
[customized version](https://github.com/bernhardpg/drake/tree/towards-tight-convex-relaxations)
of Drake. To install, follow these instructions:

Navigate to a desired installation location and run:

```console
git clone git@github.com:bernhardpg/drake.git 
cd drake
git checkout towards-tight-convex-relaxations
```

To build Drake and the python bindings, run:

```console
cd ../
mkdir drake-build
cd drake-build
cmake -DWITH_MOSEK=ON ../
make install
```

See [the docs](https://drake.mit.edu/from_source.html) for more information on building Drake.

#### Activating the environment

To activate the environment, run:

```console
poetry shell
```

Further, to point the environment to the customized Drake build, run:

```console
export PYTHONPATH={DRAKE_BUILD_DIR_PATH}/install/lib/python3.11/site-packages:${PYTHONPATH}
```

where `{DRAKE_BUILD_DIR_PATH}` should be replaced with the absolute path to the `drake-build` directory above.

---

## Generating planar pushing plans

Currently, the main entrypoint for generating planar pushing plans is the
following script:

```python
python scripts/planar_pushing/create_plan.py
```

which takes a number of command line arguments. Add the flag `--help` for a
description of these.

For instance, to generate 10 plans for a box slider geometry, run

```python
python scripts/planar_pushing/create_plan.py --body box --seed 0 --num 10
```

## Running a single hardware experiment

Create a config file specifying the experiment in `config` and run it using the
following command:

```python
python3 scripts/planar_pushing/run_planar_pushing_experiment.py --config-name single_experiment
```

where `single_experiment` should be replaced with your config name.

## (Optional) Additional packages

Make sure to have graphviz installed on your computer. On MacOS, run the following
command:

```python
brew install graphviz
```
