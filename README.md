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

Finally, make sure to have graphviz installed on your computer. On MacOS, run the following command:
```
brew install graphviz
```

Now everything needed to run the project should be installed. To run any files in the project, prefix the command with `poetry run ...`. For example:
```
poetry run python main.py
```

## Running demos
```
poetry run python main.py --demo box_push
```

```
poetry run python main.py --demo box_pickup
```

## Running pre-commit hooks
The repo is setup to do automatic linting and code checking on every commit through the use of pre-commits. To run all the pre-commit hooks (which will clean up all files in the repo), run the following command:
```
poetry run pre-commit run --all-files
```

