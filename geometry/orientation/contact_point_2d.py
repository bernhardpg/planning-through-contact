import numpy as np
import numpy.typing as npt
import pydrake.symbolic as sym  # type: ignore

from geometry.box import RigidBody2d
from geometry.contact_2d.types import ContactLocation, ContactType
from tools.types import NpExpressionArray, NpFormulaArray, NpVariableArray


class ContactPoint2d:
    def __init__(
        self,
        body: RigidBody2d,
        contact_location: ContactLocation,
        friction_coeff: float = 0.5,
        name: str = "unnamed",
    ) -> None:
        self.normal_vec, self.tangent_vec = body.get_norm_and_tang_vecs_from_location(
            contact_location
        )
        self.friction_coeff = friction_coeff
        self.contact_location = contact_location

        self.normal_force = sym.Variable(f"{name}_c_n")
        self.friction_force = sym.Variable(f"{name}_c_f")

        if self.contact_location.type == ContactType.FACE:
            self.lam = sym.Variable(f"{name}_lam")
            vertices = body.get_proximate_vertices_from_location(self.contact_location)
            self.contact_position = (
                self.lam * vertices[0] + (1 - self.lam) * vertices[1]
            )
        else:
            corner_vertex = body.get_proximate_vertices_from_location(
                self.contact_location
            )
            self.contact_position = corner_vertex

    @property
    def contact_force(self) -> NpExpressionArray:
        return (
            self.normal_force * self.normal_vec + self.friction_force * self.tangent_vec
        )

    @property
    def variables(self) -> NpVariableArray:
        if self.contact_location.type == ContactType.FACE:
            return np.array([self.normal_force, self.friction_force, self.lam])
        else:
            return np.array([self.normal_force, self.friction_force])

    def create_friction_cone_constraints(self) -> NpFormulaArray:
        upper_bound = self.friction_force <= self.friction_coeff * self.normal_force
        lower_bound = -self.friction_coeff * self.normal_force <= self.friction_force
        normal_force_positive = self.normal_force >= 0
        return np.vstack([upper_bound, lower_bound, normal_force_positive])
