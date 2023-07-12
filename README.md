# Planning Through Contact

## Installation (Linux and MacOS)
This repo uses Poetry for dependency management. To setup this project, first install [Poetry](https://python-poetry.org/docs/#installation) and, make sure to have Python3.10 installed on your system.

Then, configure poetry to setup a virtual environment that uses Python 3.10:
```
poetry env use python3.10
```

Next, install all the required dependencies to the virtual environment with the following command:
```
poetry install -vvv
```
(the `-vvv` flag adds verbose output).

## Add this library to your pythonpath
Add this module to pythonpath:

```
export PYTHONPATH=$PYTHONPATH:.
```

## Some experiments
Elimination of linear equality constraints:

```
python experiments/planar_pushing/one_contact_mode.py
```

## Running (deprecated) demos
```
poetry run python deprecated/run_demos.py --demo box_push
```

```
poetry run python deprecated/run_demos.py --demo box_pickup
```

Planning for flipping up polytopes of different shapes:
```
poetry run python experiments/object-flipup/polytope_flipup.py --num_vertices 3
poetry run python experiments/object-flipup/polytope_flipup.py --num_vertices 4
poetry run python experiments/object-flipup/polytope_flipup.py --num_vertices 5

```

## Running pre-commit hooks
The repo is setup to do automatic linting and code checking on every commit through the use of pre-commits. To run all the pre-commit hooks (which will clean up all files in the repo), run the following command:
```
poetry run pre-commit run --all-files
```

## Running hardware simulations
Make sure to download `lcm` and build it to a desired directory: [lcm](https://github.com/lcm-proj/lcm). Then, add `lcm` to the pythonpath:
```
export PYTHONPATH="$PYTHONPATH:/Users/bernhardpg/software/lcm/build/python"
```
TODO: @bernhardpg complete this

## Additional packages
Make sure to have graphviz installed on your computer. On MacOS, run the following command:
```
brew install graphviz
```
