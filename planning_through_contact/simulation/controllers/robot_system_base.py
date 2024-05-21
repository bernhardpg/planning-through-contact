from abc import abstractmethod

from pydrake.all import Diagram, Meshcat


class RobotSystemBase(Diagram):
    """The position controller base class."""

    def __init__(self):
        super().__init__()

    def add_meshcat(self, meshcat: Meshcat) -> None:
        self._meshcat = meshcat

    def pre_sim_callback(self, root_context):
        ...

    @property
    @abstractmethod
    def robot_model_name(self) -> str:
        """The name of the robot model."""
        ...

    @abstractmethod
    def num_positions(self) -> int:
        """The number of positions in the robot model."""
        ...

    # methods for visualization functions

    @property
    @abstractmethod
    def slider_model_name(self) -> str:
        """The name of the robot model."""
        ...

    @abstractmethod
    def get_station_plant(self):
        ...

    @abstractmethod
    def get_scene_graph(self):
        ...

    @abstractmethod
    def get_slider(self):
        ...

    @abstractmethod
    def get_meshcat(self):
        ...
