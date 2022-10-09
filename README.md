# Planning Through Contact

## Installation
This repo uses Poetry for dependency management. To setup this project, make sure to have both [Poetry](https://python-poetry.org/docs/#installation) and Python3.10 installed on your system.

Then, configure poetry to use Python3.10:
```
poetry env use python3.10
```

Next, install all the required dependencies with:
```
poetry install -vvv
```
(the `-vvv` flag adds verbose output).




## Administrative TODOS
- [ ] Setup pre-commits
- [ ] Setup linting
- [ ] Setup type checking

## Research plan
- [X] Formulate simple GCS for position path planning
- [ ] Replace cost and constraints by using Binding
- [ ] Formulate GCS for 1D pusher
- [ ] Formulate GCS for 2D pusher
- [ ] Integrate with Drake library features when merged
