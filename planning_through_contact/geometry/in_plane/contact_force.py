from typing import Literal, NamedTuple, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pydrake.symbolic as sym

from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    CollisionGeometry,
    ContactLocation,
    PolytopeContactLocation,
)
from planning_through_contact.tools.types import (
    NpExpressionArray,
    NpFormulaArray,
    NpVariableArray,
)


class ContactForceDefinition(NamedTuple):
    name: str
    friction_coeff: float
    location: PolytopeContactLocation
    body_geometry: CollisionGeometry
    fixed_to_friction_cone_boundary: Optional[Literal["LEFT", "RIGHT"]] = None
    displacement: float = 0

    """
    A tuple that contains all the relevant information for creating a contact force and
    its derived quantities.
    
    displacement: useful when there are multiple contact points with a fixed displacement
                  (i.e.) for a face on face contact
    """


class ContactForce(NamedTuple):
    name: str
    location: ContactLocation
    friction_coeff: float
    force: NpExpressionArray
    pos: npt.NDArray[np.float64] | NpExpressionArray
    variables: NpVariableArray
    normal_vec: Optional[npt.NDArray[np.float64]]

    def create_friction_cone_constraints(self) -> NpFormulaArray:
        if not self.location == ContactLocation.FACE:
            raise ValueError(
                "Can only create friction cone constraints for a contact force on a face. For contact forces on vertices you must rely on Newton's third law."
            )
        normal_force = self.variables[0]
        normal_force_positive = normal_force >= 0
        # TODO(bernhardpg): This is a quick fix for the case where the friction cone is fixed. Should be cleaned up!
        if len(self.variables) == 1:
            return np.array([normal_force_positive]).reshape([-1, 1])
        else:
            friction_force = self.variables[1]
            upper_bound = friction_force <= self.friction_coeff * normal_force
            lower_bound = -self.friction_coeff * normal_force <= friction_force

            return np.vstack([upper_bound, lower_bound, normal_force_positive])

    @classmethod
    def from_definition(
        cls,
        force_def: ContactForceDefinition,
        p_Bc: npt.NDArray[np.float64] | NpExpressionArray,
    ) -> "ContactForce":
        contact_force, symbolic_vars = cls._create_contact_force(force_def)

        if force_def.location.pos == ContactLocation.FACE:
            # NOTE: For face contacts, we will have multiple contact points.
            # Currently, this is modelled as having one decision variable for the position
            # of both contact points, and then displacing each contact pose by a suitable amount
            # (based on geometry of the objects) from this location.

            # TODO: In the future, to allow for face contacts which are not fully overlapping,
            # we will have one variable for each contact point.

            assert isinstance(
                p_Bc.dtype, object
            )  # Make sure we have a NpExpressionArray
            force_position = cls._create_force_position(force_def, p_Bc)  # type: ignore
            (
                normal_vec,
                _,
            ) = force_def.body_geometry.get_norm_and_tang_vecs_from_location(
                force_def.location
            )
        else:
            force_position = p_Bc
            normal_vec = None

        return cls(
            force_def.name,
            force_def.location.pos,
            force_def.friction_coeff,
            contact_force,
            force_position,
            symbolic_vars,
            normal_vec,
        )

    @staticmethod
    def _create_force_position(
        force_def: ContactForceDefinition, contact_point_position: NpExpressionArray
    ) -> NpExpressionArray:
        vertices = force_def.body_geometry.get_proximate_vertices_from_location(
            force_def.location
        )
        temp = vertices[1] - vertices[0]
        vec = temp / np.linalg.norm(temp)

        force_position = contact_point_position + vec * force_def.displacement  # type: ignore
        return force_position

    @staticmethod
    def _create_contact_force(
        force_def: ContactForceDefinition,
    ) -> Tuple[NpExpressionArray, NpVariableArray]:
        if force_def.location.pos == ContactLocation.FACE:
            # We are a face contact, hence we use the normal vector from the polytope face
            normal_force = sym.Variable(f"{force_def.name}_c_n")
            (
                normal_vec,
                tangent_vec,
            ) = force_def.body_geometry.get_norm_and_tang_vecs_from_location(
                force_def.location
            )

            if force_def.fixed_to_friction_cone_boundary is None:  # Rolling
                # Allow the friction force to be anywhere inside friction cone
                friction_force = sym.Variable(f"{force_def.name}_c_f")
                vars = [normal_force, friction_force]

            else:  # Sliding
                vars = [normal_force]
                # When we are on the friction cone, the friction force component is a linear function of the normal force
                if force_def.fixed_to_friction_cone_boundary == "LEFT":
                    friction_force = -force_def.friction_coeff * normal_force
                elif force_def.fixed_to_friction_cone_boundary == "RIGHT":
                    friction_force = force_def.friction_coeff * normal_force
                else:
                    raise ValueError("Friction cone not free, but direction not set!")

            contact_force = normal_force * normal_vec + friction_force * tangent_vec

        # For vertex contacts we rely on equal and opposite forces to satisfy friction cone constraints
        elif force_def.location.pos == ContactLocation.VERTEX:
            contact_force = np.array(
                [
                    sym.Variable(f"{force_def.name}_f_x"),
                    sym.Variable(f"{force_def.name}_f_y"),
                ]
            ).reshape((-1, 1))
            vars = contact_force.flatten()
        else:
            raise ValueError(f"Unknown contact location {force_def.location.pos}")

        return contact_force, vars  # type: ignore
