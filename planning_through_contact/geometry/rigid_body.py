from dataclasses import dataclass
from typing import Optional

from pydrake.geometry import Box as DrakeBox
from pydrake.geometry import Shape as DrakeShape
from pydrake.multibody.tree import RigidBody as DrakeRigidBody
from pydrake.multibody.tree import SpatialInertia

from planning_through_contact.geometry.collision_geometry.box_2d import Box2d
from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    CollisionGeometry,
)


@dataclass
class RigidBody:
    name: str
    geometry: CollisionGeometry
    mass: float
    is_actuated: bool = False

    @classmethod
    def from_drake(
        cls,
        shape: DrakeShape | DrakeBox,
        rigid_body: DrakeRigidBody,
        name: Optional[str] = None,
    ) -> "RigidBody":
        name = name if name else rigid_body.name()

        if not isinstance(shape, DrakeBox):
            raise NotImplementedError(
                "Only Drake box shapes are supported at the moment"
            )

        geometry = Box2d.from_drake(shape)
        spatial_inertia = rigid_body.default_spatial_inertia()
        return cls(name, geometry, spatial_inertia.get_mass(), False)
