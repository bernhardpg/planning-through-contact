from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from geometry.two_d.contact.types import ContactLocation

GRAV_ACC = 9.81


@dataclass
class RigidBody2d(ABC):
    actuated: bool
    name: str
    mass: Optional[float]

    @property
    def gravity_force_in_W(self) -> npt.NDArray[np.float64]:
        if self.mass is None:
            raise ValueError(
                "Rigid body must have a mass to calculate gravitational force"
            )
        return np.array([0, -self.mass * GRAV_ACC]).reshape((-1, 1))


class Polytope2d(RigidBody2d):
    @abstractmethod
    def get_proximate_vertices_from_location(
        self, location: ContactLocation
    ) -> List[npt.NDArray[np.float64]]:
        pass

    @abstractmethod
    def get_norm_and_tang_vecs_from_location(
        self, location: ContactLocation
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        pass


class Point2d(RigidBody2d):
    ...
