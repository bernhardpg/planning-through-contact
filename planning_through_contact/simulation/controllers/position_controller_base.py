from abc import ABC, abstractmethod
from typing import List

from pydrake.all import DiagramBuilder, MultibodyPlant, Meshcat, System


class PositionControllerBase(ABC):
    """The position controller base class."""

    @abstractmethod
    def setup(self, builder: DiagramBuilder, plant: MultibodyPlant, **kwargs) -> System:
        """Setup the position controller."""
        raise NotImplementedError

    def add_meshcat(self, meshcat: Meshcat) -> None:
        self._meshcat = meshcat
