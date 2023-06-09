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

## Installing Drake
Drake must be built from source and added to the pythonpath, after activating the virtual environment (TODO: @bernhardpg fix this, only a short term solution). See the "Building the python bindings" section in https://robotlocomotion.github.io/from_source.html.
Then run something like:
```
cd drake-build
export PYTHONPATH=${PWD}/install/lib/python3.11/site-packages:${PYTHONPATH}
```
in my case
```
export PYTHONPATH=~/software/drake-build/install/lib/python3.11/site-packages:${PYTHONPATH}
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

