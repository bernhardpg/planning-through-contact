from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional, Union

import numpy as np
import numpy.typing as npt
import pydrake.symbolic as sym
from pydrake.math import eq

from geometry.two_d.contact.contact_pair_2d import (
    AbstractContactPair,
    ContactFrameConstraints,
    ContactPairDefinition,
    LineContactConstraints,
    PairContactConstraints,
)
from geometry.two_d.contact.contact_point_2d import ContactForce, ContactPoint2d
from geometry.two_d.contact.types import ContactMode
from geometry.two_d.rigid_body_2d import RigidBody2d
from geometry.utilities import cross_2d
from tools.types import NpExpressionArray, NpFormulaArray, NpVariableArray


class FrictionConeDetails(NamedTuple):
    normal_vec_local: npt.NDArray[np.float64]
    R_WFc: NpExpressionArray
    p_WFc_W: NpExpressionArray


class StaticEquilibriumConstraints(NamedTuple):
    force_balance: NpFormulaArray
    torque_balance: NpFormulaArray

    # Rotation matrix is normalized.
    # This is to evaluate the force balance violation after the SO(2) constraints are relaxed
    normalized_force_balance: Optional[NpFormulaArray]


class ContactSceneConstraints(NamedTuple):
    pair_constraints: List[Union[LineContactConstraints, PairContactConstraints]]
    static_equilibrium_constraints: List[StaticEquilibriumConstraints]


@dataclass
class ContactScene2d:
    rigid_bodies: List[RigidBody2d]
    contact_pairs: List[ContactPairDefinition]
    body_anchored_to_W: RigidBody2d

    def create_instance(
        self,
        contact_pair_modes: Dict[str, ContactMode],
        instance_postfix: Optional[str] = None,
    ) -> "ContactSceneInstance":
        """
        Instantiates an instance of each contact pair with new sets of variables and constraints.
        """
        contact_pair_instances = [
            pair.create_instance(contact_pair_modes[pair.name], instance_postfix)
            for pair in self.contact_pairs
        ]
        return ContactSceneInstance(
            self.rigid_bodies, contact_pair_instances, self.body_anchored_to_W
        )


class ContactSceneInstance:
    def __init__(
        self,
        rigid_bodies: List[RigidBody2d],
        contact_pairs: List[AbstractContactPair],
        body_anchored_to_W: RigidBody2d,
    ):
        self.rigid_bodies = rigid_bodies
        self.contact_pairs = contact_pairs
        self.body_anchored_to_W = body_anchored_to_W

    def _get_contact_forces_acting_on_body(
        self, body: RigidBody2d
    ) -> List[NpExpressionArray]:
        contact_forces_on_body = [
            force
            for pair in self.contact_pairs
            for point in pair.contact_points
            for force in point.get_contact_forces()
            if point.body == body
        ]
        return contact_forces_on_body

    def _get_contact_points_on_body(
        self, body: RigidBody2d
    ) -> List[Union[NpExpressionArray, npt.NDArray[np.float64]]]:
        contact_points = [
            pos
            for pair in self.contact_pairs
            for point in pair.contact_points
            for pos in point.get_contact_positions()
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
            # Search for connections to the world frame ONE layer down
            # NOTE: Does not search deeper than one transformation
            R_NB = None
            next_body = None
            for pair in self.contact_pairs:
                if pair.body_B == body:
                    R_NB = pair.R_AB
                    next_body = pair.body_A
                    break
                elif pair.body_A == body:
                    R_NB = pair.R_AB.T
                    next_body = pair.body_B
                    break

            if next_body is None or R_NB is None:
                raise ValueError(
                    f"No transformation found from {body.name} to {self.body_anchored_to_W.name}"
                )
            R_WN = self._get_rotation_to_W(next_body)
            R_body_to_W = R_WN.dot(R_NB)

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
            # Search for connections to the world frame ONE layer down
            # NOTE: Does not search deeper than one transformation
            p_NB_N = None
            next_body = None
            for pair in self.contact_pairs:
                if pair.body_B == body:
                    p_NB_N = pair.p_AB_A
                    next_body = pair.body_A
                    break
                elif pair.body_A == body:
                    p_NB_N = pair.p_BA_B
                    next_body = pair.body_B
                    break

            if next_body is None or p_NB_N is None:
                raise ValueError(
                    f"No transformation found from {body.name} to {self.body_anchored_to_W.name}"
                )
            R_WN = self._get_rotation_to_W(next_body)
            p_NB_W = R_WN.dot(p_NB_N)

            p_WN_W = self._get_translation_to_W(next_body)
            p_WB = p_WN_W + p_NB_W

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
    def pair_constraints(
        self,
    ) -> List[Union[LineContactConstraints, PairContactConstraints]]:
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

    def get_squared_forces_for_bodies(
        self, only_unactuated_bodies: bool = False
    ) -> sym.Expression:
        bodies_to_use = (
            self.unactuated_bodies if only_unactuated_bodies else self.rigid_bodies
        )
        forces = [
            pair.get_squared_contact_forces_for_body(body)
            for pair in self.contact_pairs
            for body in bodies_to_use
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
    def __init__(
        self,
        contact_scene: ContactScene2d,
        contact_modes: Dict[str, ContactMode],
        idx: int,
    ):
        self.contact_scene_instance = contact_scene.create_instance(contact_modes, None)

    def get_gravitational_forces_in_world_frame(self) -> List[npt.NDArray[np.float64]]:
        """
        Returns the gravitational forces for all the unactuated objects in the scene.
        """
        gravitational_forces = [
            body.gravity_force_in_W
            for body in self.contact_scene_instance.unactuated_bodies
        ]
        return gravitational_forces

    def get_contact_forces_in_world_frame(self) -> List[NpExpressionArray]:
        forces_W = []
        for pair in self.contact_scene_instance.contact_pairs:
            for point in pair.contact_points:
                R_WB = self.contact_scene_instance._get_rotation_to_W(point.body)
                forces = point.get_contact_forces()
                f_cB_Ws = [R_WB.dot(f) for f in forces]
                forces_W.extend(f_cB_Ws)
        return forces_W

    def get_body_orientations(self) -> List[NpExpressionArray]:
        _convert_to_expr = np.vectorize(
            lambda x: sym.Expression(x) if not isinstance(x, sym.Expression) else x
        )  # make sure we only return expressions
        Rs = [
            _convert_to_expr(self.contact_scene_instance._get_rotation_to_W(body))
            for body in self.contact_scene_instance.rigid_bodies
        ]
        return Rs

    def get_contact_point_orientations(self) -> List[NpExpressionArray]:
        Rs = [
            self.contact_scene_instance._get_rotation_to_W(point.body)
            for pair in self.contact_scene_instance.contact_pairs
            for point in pair.contact_points
        ]
        return Rs

    def get_contact_positions_for_contact_point_in_world_frame(
        self, point: ContactPoint2d
    ) -> List[NpExpressionArray]:
        R_WB = self.contact_scene_instance._get_rotation_to_W(point.body)
        contact_positions = point.get_contact_positions()
        p_Bc1_Ws = [R_WB.dot(pos) for pos in contact_positions]
        p_WB = self.contact_scene_instance._get_translation_to_W(point.body)
        p_Wc1_Ws = [p_WB + p_Bc1_W for p_Bc1_W in p_Bc1_Ws]
        return p_Wc1_Ws

    def get_contact_positions_in_world_frame(self) -> List[NpExpressionArray]:
        pos_W = []
        for pair in self.contact_scene_instance.contact_pairs:
            for point in pair.contact_points:
                p_Wc1_Ws = self.get_contact_positions_for_contact_point_in_world_frame(
                    point
                )
                pos_W.extend(p_Wc1_Ws)
        return pos_W

    def get_body_positions_in_world_frame(self) -> List[NpExpressionArray]:
        pos_W = []
        for body in self.contact_scene_instance.rigid_bodies:
            p_WB = self.contact_scene_instance._get_translation_to_W(body)
            _convert_to_expr = np.vectorize(
                lambda x: sym.Expression(x) if not isinstance(x, sym.Expression) else x
            )  # make sure we only return expressions
            pos_W.append(_convert_to_expr(p_WB))
        return pos_W

    def _get_friction_cone_details(
        self, point: ContactPoint2d, force: ContactForce
    ) -> List[FrictionConeDetails]:
        normal_vec = force.normal_vec
        if normal_vec is None:
            raise ValueError("Could not get normal vector for force")
        R_WFc = self.contact_scene_instance._get_rotation_to_W(point.body)
        p_WFc_Ws = self.get_contact_positions_for_contact_point_in_world_frame(point)
        friction_cone_details = [
            FrictionConeDetails(normal_vec, R_WFc, p_WFc_W) for p_WFc_W in p_WFc_Ws
        ]
        return friction_cone_details

    def get_friction_cones_details_for_face_contact_points(
        self,
    ) -> List[FrictionConeDetails]:
        contact_points_on_faces = [
            pair.get_nonfixed_contact_point()
            for pair in self.contact_scene_instance.contact_pairs
        ]
        return [
            details
            for point in contact_points_on_faces
            for force in point.contact_forces
            for details in self._get_friction_cone_details(point, force)
            if force.normal_vec is not None
        ]

    @property
    def static_equilibrium_constraints(self) -> List[StaticEquilibriumConstraints]:
        return self.constraints.static_equilibrium_constraints

    @property
    def friction_cone_constraints(self) -> NpFormulaArray:
        return np.concatenate(
            [c.friction_cone for c in self.constraints.pair_constraints]
        )

    @property
    def so_2_constraints(self) -> NpFormulaArray:
        return np.array([c.so_2 for c in self.point_on_line_contact_constraints])

    @property
    def rotation_bounds(self) -> NpFormulaArray:
        return np.array(
            [c.rotation_bounds for c in self.point_on_line_contact_constraints]
        )

    @property
    def convex_hull_bounds(self) -> NpFormulaArray:
        return np.array(
            [c.convex_hull_bounds for c in self.point_on_line_contact_constraints]
        )

    @property
    def relaxed_so_2_constraints(self) -> NpFormulaArray:
        return np.array(
            [c.relaxed_so_2 for c in self.point_on_line_contact_constraints]
        )

    @property
    def non_penetration_cuts(self) -> NpFormulaArray:
        return np.array(
            [c.non_penetration_cut for c in self.point_on_line_contact_constraints]
        )

    @property
    def equal_contact_point_constraints(self) -> List[ContactFrameConstraints]:
        return [c.equal_contact_points for c in self.point_on_line_contact_constraints]

    @property
    def equal_rel_position_constraints(self) -> List[ContactFrameConstraints]:
        return [
            c.equal_relative_positions for c in self.point_on_line_contact_constraints
        ]

    @property
    def equal_and_opposite_forces_constraints(self) -> List[ContactFrameConstraints]:
        return [
            c.equal_and_opposite_forces for c in self.point_on_line_contact_constraints
        ]

    @property
    def variables(self) -> NpVariableArray:
        return self.contact_scene_instance.variables

    def get_squared_forces(
        self, only_unactuated_bodies: bool = False
    ) -> sym.Expression:
        return self.contact_scene_instance.get_squared_forces_for_bodies(
            only_unactuated_bodies
        )

    @property
    def constraints(self) -> ContactSceneConstraints:
        return self.contact_scene_instance.create_contact_scene_constraints()

    @property
    def point_on_line_contact_constraints(self) -> List[PairContactConstraints]:
        return [
            constraints
            for constraints in self.constraints.pair_constraints
            if isinstance(constraints, PairContactConstraints)
        ]
