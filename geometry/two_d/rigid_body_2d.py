from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, NamedTuple, Optional, Tuple

import numpy as np
import numpy.typing as npt

from geometry.hyperplane import Hyperplane
from geometry.two_d.contact.types import ContactPosition

GRAV_ACC = 9.81


class PolytopeContactLocation(NamedTuple):
    pos: ContactPosition
    idx: int


@dataclass
class RigidBody2d(ABC):
    actuated: bool
    name: str
    mass: Optional[float]

    @abstractmethod
    def get_proximate_vertices_from_location(
        self, location: PolytopeContactLocation
    ) -> List[npt.NDArray[np.float64]]:
        pass

    @abstractmethod
    def get_neighbouring_vertices(
        self, location: PolytopeContactLocation
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        pass

    @abstractmethod
    def get_hyperplane_from_location(
        self, location: PolytopeContactLocation
    ) -> Hyperplane:
        pass

    @abstractmethod
    def get_norm_and_tang_vecs_from_location(
        self, location: PolytopeContactLocation
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        pass

    @property
    @abstractmethod
    def vertices_for_plotting(self) -> npt.NDArray[np.float64]:
        pass

    @property
    def gravity_force_in_W(self) -> npt.NDArray[np.float64]:
        if self.mass is None:
            raise ValueError(
                "Rigid body must have a mass to calculate gravitational force"
            )
        return np.array([0, -self.mass * GRAV_ACC]).reshape((-1, 1))
