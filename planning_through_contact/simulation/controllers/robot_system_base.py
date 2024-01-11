from abc import abstractmethod

from pydrake.all import Meshcat, Diagram


class RobotSystemBase(Diagram):
    """The position controller base class."""

    def __init__(self):
        super().__init__()

    def add_meshcat(self, meshcat: Meshcat) -> None:
        self._meshcat = meshcat

    @property
    @abstractmethod
    def robot_model_name(self) -> str:
        """The name of the robot model."""
        ...
