from abc import ABC, abstractmethod

from pydrake.all import DiagramBuilder, Meshcat, OutputPort


class DesiredPositionSourceBase(ABC):
    """The desired position source base class."""

    @abstractmethod
    def AddToBuilder(self, builder: DiagramBuilder, **kwargs) -> OutputPort:
        """Setup the desired position source system."""
        raise NotImplementedError

    def add_meshcat(self, meshcat: Meshcat) -> None:
        self._meshcat = meshcat
