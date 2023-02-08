from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional, Union

import numpy as np
import numpy.typing as npt
import pydrake.symbolic as sym
from pydrake.math import eq

from geometry.two_d.contact.contact_pair_2d import (
    ContactFrameConstraints,
    ContactPair2d,
    ContactPair2dInstance,
    ContactPairConstraints,
)
from geometry.two_d.contact.types import ContactMode
from geometry.two_d.rigid_body_2d import RigidBody2d
from geometry.utilities import cross_2d
from tools.types import NpExpressionArray, NpFormulaArray, NpVariableArray


class StaticEquilibriumConstraints(NamedTuple):
    force_balance: NpFormulaArray
    torque_balance: NpFormulaArray

    # Rotation matrix is normalized.
    # This is to evaluate the force balance violation after the SO(2) constraints are relaxed
    normalized_force_balance: Optional[NpFormulaArray]


class ContactSceneConstraints(NamedTuple):
    pair_constraints: List[ContactPairConstraints]
    static_equilibrium_constraints: List[StaticEquilibriumConstraints]


@dataclass
class ContactScene2d:
    rigid_bodies: List[RigidBody2d]
    contact_pairs: List[ContactPair2d]
    body_anchored_to_W: RigidBody2d

    def create_instance(
        self, contact_pair_modes: Dict[str, ContactMode]
    ) -> "ContactSceneInstance":
        """
        Instantiates an instance of each contact pair with new sets of variables and constraints.
        """
        contact_pair_instances = [
            pair.create_instance(contact_pair_modes[pair.name])
            for pair in self.contact_pairs
        ]
        return ContactSceneInstance(
            self.rigid_bodies, contact_pair_instances, self.body_anchored_to_W
        )


class ContactSceneInstance:
    def __init__(
        self,
        rigid_bodies: List[RigidBody2d],
        contact_pairs: List[ContactPair2dInstance],
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

    def _get_rotation_to_W(self, body: RigidBody2d) -> NpExpressionArray:
        R_body_to_W = None
        if body == self.body_anchored_to_W:
            R_body_to_W = np.eye(2)

        for pair in self.contact_pairs:
            if pair.body_A == self.body_anchored_to_W and pair.body_B == body:
                R_body_to_W = pair.R_AB
            elif pair.body_B == self.body_anchored_to_W and pair.body_A == body:
                R_body_to_W = pair.R_AB.T

        if R_body_to_W is None:
            raise ValueError(
                f"No transformation found from {body.name} to {self.body_anchored_to_W.name}"
            )
        else:
            # For now we don't care about the type being float when we return identity
            return R_body_to_W  # type: ignore

    def _get_translation_to_W(self, body: RigidBody2d) -> NpExpressionArray:
        p_WB = None
        if body == self.body_anchored_to_W:
            p_WB = np.zeros((2, 1))

        for pair in self.contact_pairs:
            if pair.body_A == self.body_anchored_to_W and pair.body_B == body:
                p_WB = pair.p_AB_A
            elif pair.body_B == self.body_anchored_to_W and pair.body_A == body:
                p_WB = pair.p_BA_B

        if p_WB is None:
            raise ValueError(
                f"No transformation found from {body.name} to {self.body_anchored_to_W.name}"
            )
        else:
            # For now we don't care about the type being float when we return identity
            return p_WB  # type: ignore

    @property
    def unactuated_bodies(self) -> List[RigidBody2d]:
        bodies = [body for body in self.rigid_bodies if not body.actuated]
        return bodies

    @property
    def variables(self) -> NpVariableArray:
        return np.concatenate([pair.variables for pair in self.contact_pairs])

    @property
    def pair_constraints(self) -> List[ContactPairConstraints]:
        return [pair.create_constraints() for pair in self.contact_pairs]

    def create_static_equilibrium_constraints_for_body(
        self, body: RigidBody2d
    ) -> StaticEquilibriumConstraints:
        """
        Creates the static equilibrium constraints for 'body' in the frame of 'body'
        """

        contact_forces = self._get_contact_forces_acting_on_body(body)
        R_BW = self._get_rotation_to_W(body).T

        R_BW_normalized = R_BW * (1 / sym.sqrt(R_BW[:, 0].dot(R_BW[:, 0])))  # type: ignore

        gravity_force = R_BW.dot(body.gravity_force_in_W)
        sum_of_forces = np.sum(contact_forces, axis=0) + gravity_force
        force_balance_condition = eq(sum_of_forces, 0)

        normalized_force_balance = eq(
            np.sum(contact_forces, axis=0)
            + R_BW_normalized.dot(body.gravity_force_in_W),
            0,
        )

        contact_points = self._get_contact_points_on_body(body)
        contact_torques = [
            cross_2d(p_ci, f_ci) for p_ci, f_ci in zip(contact_points, contact_forces)
        ]
        sum_of_contact_torqes = np.array(
            [np.sum(contact_torques, axis=0)]
        )  # Convert to np array so this works with downstream functions
        torque_balance_condition = eq(sum_of_contact_torqes, 0)

        return StaticEquilibriumConstraints(
            force_balance_condition, torque_balance_condition, normalized_force_balance
        )

    @property
    def static_equilibrium_constraints(self) -> List[StaticEquilibriumConstraints]:
        return [
            self.create_static_equilibrium_constraints_for_body(body)
            for body in self.unactuated_bodies
        ]

    def get_squared_forces_for_unactuated_bodies(self) -> sym.Expression:
        forces = [
            pair.get_squared_contact_forces_for_body(body)
            for pair in self.contact_pairs
            for body in self.unactuated_bodies
            if body in pair.bodies
        ]
        sum_of_forces = np.sum(forces, axis=0)[0, 0]  # type: ignore
        return sum_of_forces

    def create_contact_scene_constraints(self) -> ContactSceneConstraints:
        return ContactSceneConstraints(
            self.pair_constraints,
            self.static_equilibrium_constraints,
        )


class ContactSceneCtrlPoint:
    def __init__(self, contact_scene_instance: ContactSceneInstance):
        self.contact_scene_instance = contact_scene_instance

        self.table_box = self.contact_scene_instance.contact_pairs[0]
        self.box_finger = self.contact_scene_instance.contact_pairs[1]

        self.table = self.contact_scene_instance.rigid_bodies[0]
        self.box = self.contact_scene_instance.rigid_bodies[1]
        self.finger = self.contact_scene_instance.rigid_bodies[2]

        # Define convenience variables for plotting
        self.p_TB_T = self.table_box.p_AB_A
        self.p_WB_W = self.p_TB_T

        self.p_BT_B = self.table_box.p_BA_B
        self.p_BW_B = self.p_BT_B

        self.R_TB = self.table_box.R_AB
        self.R_WB = self.R_TB  # World frame is the same as table frame

        self.fc1_B = self.table_box.contact_point_B.contact_force
        self.fc1_T = self.table_box.contact_point_A.contact_force
        self.pc1_B = self.table_box.contact_point_B.contact_position
        self.pc1_T = self.table_box.contact_point_A.contact_position

        self.fc2_B = self.box_finger.contact_point_A.contact_force
        self.fc2_F = self.box_finger.contact_point_B.contact_force
        self.pc2_B = self.box_finger.contact_point_A.contact_position
        self.pc2_F = self.box_finger.contact_point_B.contact_position

    def get_contact_forces_in_world_frame(self) -> List[NpExpressionArray]:
        forces_W = []
        pair = self.contact_scene_instance.contact_pairs[0]
        # for pair in self.contact_scene_instance.contact_pairs: # FIX: Comment in after I add support for one sided contacts!
        for point in pair.contact_points:
            R_WB = self.contact_scene_instance._get_rotation_to_W(point.body)
            f_cB_W = R_WB.dot(point.contact_force)
            forces_W.append(f_cB_W)
        return forces_W

    def get_contact_positions_in_world_frame(self) -> List[NpExpressionArray]:
        pos_W = []
        pair = self.contact_scene_instance.contact_pairs[0]
        # for pair in self.contact_scene_instance.contact_pairs: # FIX: Comment in after I add support for one sided contacts!
        for point in pair.contact_points:
            R_WB = self.contact_scene_instance._get_rotation_to_W(point.body)
            p_Bc1_W = R_WB.dot(point.contact_position)
            p_WB = self.contact_scene_instance._get_translation_to_W(point.body)
            p_Wc1_W = p_WB + p_Bc1_W
            pos_W.append(p_Wc1_W)
        return pos_W

    @property
    def static_equilibrium_constraints(self) -> List[StaticEquilibriumConstraints]:
        return self.constraints.static_equilibrium_constraints

    @property
    def friction_cone_constraints(self) -> NpFormulaArray:
        return np.concatenate(
            [c.friction_cone for c in self.constraints.pair_constraints]
        )

    @property
    def relaxed_so_2_constraints(self) -> NpFormulaArray:
        return np.array([c.relaxed_so_2 for c in self.constraints.pair_constraints])

    @property
    def non_penetration_cuts(self) -> NpFormulaArray:
        return np.array(
            [c.non_penetration_cut for c in self.constraints.pair_constraints]
        )

    @property
    def equal_contact_point_constraints(self) -> List[ContactFrameConstraints]:
        return [c.equal_contact_points for c in self.constraints.pair_constraints]

    @property
    def equal_rel_position_constraints(self) -> List[ContactFrameConstraints]:
        return [c.equal_relative_positions for c in self.constraints.pair_constraints]

    @property
    def equal_and_opposite_forces_constraints(self) -> List[ContactFrameConstraints]:
        return [c.equal_and_opposite_forces for c in self.constraints.pair_constraints]

    @property
    def variables(self) -> NpVariableArray:
        return self.contact_scene_instance.variables

    @property
    def squared_forces(self) -> sym.Expression:
        return self.contact_scene_instance.get_squared_forces_for_unactuated_bodies()

    @property
    def constraints(self) -> ContactSceneConstraints:
        return self.contact_scene_instance.create_contact_scene_constraints()
