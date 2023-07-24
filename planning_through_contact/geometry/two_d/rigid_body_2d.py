from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, NamedTuple, Optional, Tuple

import numpy as np
import numpy.typing as npt

from planning_through_contact.geometry.hyperplane import Hyperplane
from planning_through_contact.geometry.two_d.contact.types import ContactLocation

GRAV_ACC = 9.81

# TODO: Deprecate this in favor of new, more general rigid_body using Drake functionality


class PolytopeContactLocation(NamedTuple):
    pos: ContactLocation
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

    def get_shortest_vec_from_com_to_face(
        self, location: PolytopeContactLocation
    ) -> npt.NDArray[np.float64]:
        v1, v2 = self.get_proximate_vertices_from_location(location)
        vec = (v1 + v2) / 2
        return vec

    @abstractmethod
    def get_face_length(self, location: PolytopeContactLocation) -> float:
        pass
