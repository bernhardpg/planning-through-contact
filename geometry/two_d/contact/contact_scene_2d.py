from typing import List, NamedTuple, Union

import numpy as np
import numpy.typing as npt
from pydrake.math import eq

from geometry.two_d.contact.contact_pair_2d import ContactPair2d
from geometry.two_d.rigid_body_2d import RigidBody2d
from geometry.utilities import cross_2d
from tools.types import NpExpressionArray, NpFormulaArray, NpVariableArray


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

    def _get_contact_points_on_body(
        self, body: RigidBody2d
    ) -> List[Union[NpExpressionArray, npt.NDArray[np.float64]]]:
        contact_points = [
            point.contact_position
            for pair in self.contact_pairs
            for point in pair.contact_points
            if point.body == body
        ]
        return contact_points

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

    @property
    def variables(self) -> NpVariableArray:
        return np.concatenate([pair.variables for pair in self.contact_pairs])

    def create_static_equilibrium_constraints_for_body(
        self, body: RigidBody2d
    ) -> StaticEquilibriumConstraints:
        """
        Creates the static equilibrium constraints for 'body' in the frame of 'body'
        """

        contact_forces = self._get_contact_forces_acting_on_body(body)
        R_BW = self._get_transformation_to_W(body).T

        gravity_force = R_BW.dot(body.gravity_force_in_W)
        sum_of_forces = np.sum(contact_forces, axis=0) + gravity_force
        force_balance_condition = eq(sum_of_forces, 0)

        contact_points = self._get_contact_points_on_body(body)
        contact_torques = [
            cross_2d(p_ci, f_ci) for p_ci, f_ci in zip(contact_points, contact_forces)
        ]
        torque_balance_condition = eq(np.sum(contact_torques, axis=0), 0)

        return StaticEquilibriumConstraints(
            force_balance_condition, torque_balance_condition
        )
