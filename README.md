# Planning Through Contact

## RSS 2024: Towards Tight Convex Relaxations for Contact-Rich Manipulation

**🌐 Paper website**: [Towards Tight Convex Relaxations for Contact-Rich Manipulation](https://bernhardgraesdal.com/rss24-towards-tight-convex-relaxations/)

**📋 Paper**: [arXiv](https://arxiv.org/pdf/2402.10312)

<p align="center">
  <img src="images/demo_triangle.gif" alt="Demo triangle" width="30%" />
  <img src="images/demo_convex4.gif" alt="Demo convex object" width="30%" />
  <img src="images/demo_tee.gif" alt="Demo T" width="30%" />
</p>

📘 You can find an introduction notebook to generate planar pushing trajectories [here](https://github.com/bernhardpg/planning-through-contact/blob/introductory-notebook/notebooks/planar_pushing.ipynb).

**If you run into any problems, don't hesitate to reach out or create an issue.**

### Paper results

💻 The exact code used for generating the results in the paper can be found on the branch: [rss24-towards-tight-convex](https://github.com/bernhardpg/planning-through-contact/tree/rss24-towards-tight-convex).
The trajectories and data in the paper are generated with `scripts/planar_pushing/create_plans.py`, see the user guide below.
However, if you are simply interested in using the code, it is recommended to use the updated code on the `main` branch.

---

**⚠️ Note: This repo is under active development changes to the code may occur. ⚠️**

## 🚀 Installation (Linux and MacOS)

This repo uses Poetry for dependency management. To setup this project, first
install [Poetry](https://python-poetry.org/docs/#installation) and, make sure
to have Python3.12 installed on your system.

Then, configure poetry to setup a virtual environment that uses Python 3.12:

```python
poetry env use python3.12
```

Next, install all the required dependencies to the virtual environment with the
following command:

```python
poetry install -vvv
```

(the `-vvv` flag adds verbose output).

For debug features to work, make sure to have graphviz installed on your computer. On MacOS, run the following
command:

```python
brew install graphviz
```

### Activating the environment

To activate the environment, run:

```console
poetry shell
```

### Installing customized Drake version

**⚠️ At the moment, the code relies on a custom build of Drake. This will only be the case for another week or so, meaning that this step will not be necessary very soon⚠️**

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
cmake -DWITH_MOSEK=ON -DWITH_SNOPT=ON ../drake
make install
```

if you are a member of the RLG, run:

```console
cd ../
mkdir drake-build
cd drake-build
cmake -DWITH_MOSEK=ON -DWITH_ROBOTLOCOMOTION_SNOPT=ON ../drake
make install
```

See [the docs](https://drake.mit.edu/from_source.html) for more information on building Drake.

To tell the environment to use the customized Drake build, run:

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

For instance, to generate 10 plans for a rectangular slider geometry, run

```python
python scripts/planar_pushing/create_plan.py --body sugar_box --seed 0 --num 10
```

which will generate plans that look like this:

<p align="center">
  <img src="images/demo_box.gif" alt="Demo triangle" width="70%" />
</p>

# Developers

Before merging a branch to `main`, make sure the unit test pass by running

```
pytest
```

(after activating the virtual environment).

# Other (experimental)

## Running a single hardware experiment

Create a config file specifying the experiment in `config` and run it using the
following command:

```python
python3 scripts/planar_pushing/run_planar_pushing_experiment.py --config-name single_experiment
```

where `single_experiment` should be replaced with your config name.
