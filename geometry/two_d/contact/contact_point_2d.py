from typing import Union

import numpy as np
import numpy.typing as npt
import pydrake.symbolic as sym  # type: ignore

from geometry.two_d.contact.types import ContactLocation, ContactType
from geometry.two_d.rigid_body_2d import Point2d, Polytope2d, RigidBody2d
from tools.types import NpExpressionArray, NpFormulaArray, NpVariableArray


class ContactPoint2d:
    def __init__(
        self,
        body: RigidBody2d,
        other_body: RigidBody2d,
        contact_location: ContactLocation,
        other_contact_location: ContactLocation,
        friction_coeff: float = 0.5,
        name: str = "unnamed",
    ) -> None:
        """
        Implements a contact point with all the corresponding contact constriants defined in the local frame of 'body'.
        """

        self.name = name
        self.body = body
        self.friction_coeff = friction_coeff
        self.contact_location = contact_location

        self.normal_force = sym.Variable(f"{self.name}_c_n")
        self.friction_force = sym.Variable(f"{self.name}_c_f")

        if isinstance(body, Point2d):
            # Use the normal and tangent vector of the contact body if we are dealing with a point contact
            (
                self.normal_vec,
                self.tangent_vec,
            ) = other_body.get_norm_and_tang_vecs_from_location(  # type: ignore
                other_contact_location
            )
        elif isinstance(body, Polytope2d):
            (
                self.normal_vec,
                self.tangent_vec,
            ) = body.get_norm_and_tang_vecs_from_location(contact_location)
        else:
            raise NotImplementedError(f"Body type {body} not supported")

        self.contact_position = self._set_contact_position()

    def _set_contact_position(
        self,
    ) -> Union[npt.NDArray[np.float64], NpExpressionArray]:
        if isinstance(self.body, Polytope2d):
            if self.contact_location.type == ContactType.FACE:
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
        elif isinstance(self.body, Point2d):
            return np.array([[0], [0]])
        else:
            raise NotImplementedError(f"Body of type {self.body} not supported.")

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
