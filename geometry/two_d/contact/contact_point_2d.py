from typing import List, Literal, NamedTuple, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import pydrake.symbolic as sym  # type: ignore

from geometry.hyperplane import Hyperplane
from geometry.two_d.box_2d import RigidBody2d
from geometry.two_d.contact.types import ContactLocation
from geometry.two_d.rigid_body_2d import PolytopeContactLocation
from tools.types import NpExpressionArray, NpFormulaArray, NpVariableArray


class ContactForceDefinition(NamedTuple):
    name: str
    friction_coeff: float
    location: PolytopeContactLocation
    body: RigidBody2d
    fixed_to_friction_cone_boundary: Optional[Literal["LEFT", "RIGHT"]] = None
    displacement: float = 0


class ContactForce(NamedTuple):
    name: str
    location: ContactLocation
    friction_coeff: float
    force: NpExpressionArray
    pos: NpExpressionArray
    variables: NpVariableArray
    normal_vec: Optional[npt.NDArray[np.float64]]

    def create_friction_cone_constraints(self) -> NpFormulaArray:
        if not self.location == ContactLocation.FACE:
            raise ValueError(
                "Can only create friction cone constraints for a contact force on a face. For contact forces on vertices you must rely on Newton's third law."
            )
        normal_force = self.variables[0]
        normal_force_positive = normal_force >= 0
        # FIX: This is a quick fix for the case where the friction cone is fixed. Should be cleaned up!
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
        contact_point_position: NpExpressionArray,
    ) -> "ContactForce":
        contact_force, symbolic_vars = cls._create_contact_force(force_def)
        if force_def.location.pos == ContactLocation.FACE:
            force_position = cls._create_force_position(
                force_def, contact_point_position
            )
            normal_vec, _ = force_def.body.get_norm_and_tang_vecs_from_location(
                force_def.location
            )
        else:
            force_position = contact_point_position
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
        vertices = force_def.body.get_proximate_vertices_from_location(
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
            ) = force_def.body.get_norm_and_tang_vecs_from_location(force_def.location)

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


class ContactPoint2d:
    def __init__(
        self,
        body: RigidBody2d,
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
        self.body = body
        self.friction_coeff = friction_coeff
        self.contact_location = contact_location

        self._contact_position = self._set_contact_position()
        self.contact_forces = [ContactForce.from_definition(d, self._contact_position) for d in contact_force_defs]  # type: ignore

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
            vertices = self.body.get_proximate_vertices_from_location(
                self.contact_location
            )
            return self.lam * vertices[0] + (1 - self.lam) * vertices[1]
        else:
            # Get first element as we know this will only be one vertex
            corner_vertex = self.body.get_proximate_vertices_from_location(
                self.contact_location
            )[0]
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
        return self.body.get_neighbouring_vertices(self.contact_location)

    def get_contact_hyperplane(
        self,
    ) -> Hyperplane:
        return self.body.get_hyperplane_from_location(self.contact_location)
