from typing import List, Tuple, Union

import numpy as np
import numpy.typing as npt
import pydrake.symbolic as sym

from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    CollisionGeometry,
    ContactLocation,
    PolytopeContactLocation,
)
from planning_through_contact.geometry.hyperplane import Hyperplane
from planning_through_contact.geometry.in_plane.contact_force import (
    ContactForce,
    ContactForceDefinition,
)
from planning_through_contact.tools.types import (
    NpExpressionArray,
    NpFormulaArray,
    NpVariableArray,
)


class ContactPoint:
    def __init__(
        self,
        collision_geometry: CollisionGeometry,
        contact_location: PolytopeContactLocation,
        contact_force_defs: List[ContactForceDefinition],
        friction_coeff: float = 0.5,
        name: str = "unnamed",
    ) -> None:
        """
        Definition of a contact point.

        A contact point can have one or more forces associated with it. If it is a vertex contact point, it will have one force associated with it.
        If it is a contact point on a face, it can have two contact forces associated with it at a constant displacement from the contact point.
        """
        self.name = name
        self.collision_geometry = collision_geometry
        self.friction_coeff = friction_coeff
        self.contact_location = contact_location

        self.contact_position = self._set_contact_position()
        self.contact_forces = [ContactForce.from_definition(d, self.contact_position) for d in contact_force_defs]  # type: ignore

    def get_contact_positions(
        self,
    ) -> List[Union[NpExpressionArray, npt.NDArray[np.float64]]]:
        return [f.pos for f in self.contact_forces]

    def get_contact_forces(self) -> List[NpExpressionArray]:
        return [f.force for f in self.contact_forces]

    def _set_contact_position(
        self,
    ) -> Union[npt.NDArray[np.float64], NpExpressionArray]:
        if self.contact_location.pos == ContactLocation.FACE:
            self.lam = sym.Variable(f"{self.name}_lam")
            vertices = self.collision_geometry.get_proximate_vertices_from_location(
                self.contact_location
            )
            return self.lam * vertices[0] + (1 - self.lam) * vertices[1]
        else:
            # Get first element as we know this will only be one vertex
            corner_vertex = (
                self.collision_geometry.get_proximate_vertices_from_location(
                    self.contact_location
                )[0]
            )
            return corner_vertex

    @property
    def contact_force(self) -> NpExpressionArray:
        if len(self.contact_forces) > 1:
            raise ValueError(
                "Can only get unique contact force when there is one contact force in the contact point!"
            )

        return self.contact_forces[0].force

    @property
    def variables(self) -> NpVariableArray:
        force_vars = np.array([var for f in self.contact_forces for var in f.variables])
        if self.contact_location.pos == ContactLocation.FACE:
            return np.concatenate([force_vars, [self.lam]])  # type: ignore
        else:
            return force_vars

    def create_friction_cone_constraints(self) -> NpFormulaArray:
        friction_cone_constraints = []
        for force in self.contact_forces:
            if not force.location == ContactLocation.FACE:
                raise ValueError(
                    "Can only create friction cone constraints for a contact force on a face."
                )
            friction_cone_constraints.append(force.create_friction_cone_constraints())
        return np.concatenate(friction_cone_constraints)

    def get_neighbouring_vertices(
        self,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        return self.collision_geometry.get_neighbouring_vertices(self.contact_location)

    def get_contact_hyperplane(
        self,
    ) -> Hyperplane:
        return self.collision_geometry.get_hyperplane_from_location(
            self.contact_location
        )
