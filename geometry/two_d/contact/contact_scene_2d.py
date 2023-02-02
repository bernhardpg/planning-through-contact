from typing import List, NamedTuple

from geometry.two_d.contact.contact_pair_2d import ContactPair2d
from geometry.two_d.rigid_body_2d import RigidBody2d
from tools.types import NpExpressionArray, NpFormulaArray
from pydrake.math import eq


class StaticEquilibriumConstraints(NamedTuple):
    force_balance: NpFormulaArray
    torque_balance: NpFormulaArray


class ContactScene2d:
    def __init__(
        self,
        rigid_bodies: List[RigidBody2d],
        contact_pairs: List[ContactPair2d],
        body_anchored_to_W: RigidBody2d,
    ):
        self.rigid_bodies = rigid_bodies
        self.contact_pairs = contact_pairs
        self.body_anchored_to_W = body_anchored_to_W

    def _get_contact_forces_acting_on_body(
        self, body: RigidBody2d
    ) -> List[NpExpressionArray]:
        contact_forces_on_body = [
            point.contact_force
            for pair in self.contact_pairs
            for point in pair.contact_points
            if point.body == body
        ]
        return contact_forces_on_body

    def _get_transformation_to_W(self, body: RigidBody2d) -> NpExpressionArray:
        R_body_to_W = None
        for pair in self.contact_pairs:
            if pair.body_A == self.body_anchored_to_W and pair.body_B == body:
                R_body_to_W = pair.R_AB
            elif pair.body_B == self.body_anchored_to_W and pair.body_A == body:
                R_body_to_W = pair.R_AB.T

        if R_body_to_W is None:
            raise ValueError(
                f"No transformation found from {body} to {self.body_anchored_to_W}"
            )
        else:
            return R_body_to_W

    @property
    def unactuated_bodies(self) -> List[RigidBody2d]:
        bodies = [body for body in self.rigid_bodies if not body.actuated]
        return bodies

    def create_static_equilibrium_constraints(self) -> StaticEquilibriumConstraints:
        contact_forces_on_unactuated_bodies = [
            self._get_contact_forces_acting_on_body(body)
            for body in self.unactuated_bodies
        ]
        transformations_from_bodies_to_W = [
            self._get_transformation_to_W(body) for body in self.unactuated_bodies
        ]
        graviational_forces_in_body_frames = [
            R_WB.dot(body.gravity_force_in_W)
            for R_WB, body in zip(
                transformations_from_bodies_to_W, self.unactuated_bodies
            )
        ]
        sums_of_forces = [
            sum(contact_forces) + gravity_force
            for contact_forces, gravity_force in zip(
                contact_forces_on_unactuated_bodies, graviational_forces_in_body_frames
            )
        ]
        force_balance_conditions = {
            body.name: eq(sum_of_forces, 0)
            for body, sum_of_forces in zip(self.unactuated_bodies, sums_of_forces)
        }
        breakpoint()
