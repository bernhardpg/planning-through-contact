from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional, Union

import numpy as np
import numpy.typing as npt
import pydrake.symbolic as sym
from pydrake.math import eq

from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    ContactMode,
)
from planning_through_contact.geometry.in_plane.contact_force import ContactForce
from planning_through_contact.geometry.in_plane.contact_pair import (
    AbstractContactPair,
    ContactFrameConstraints,
    ContactPairDefinition,
    LineContactConstraints,
    PointContactConstraints,
)
from planning_through_contact.geometry.in_plane.contact_point import ContactPoint
from planning_through_contact.geometry.rigid_body import RigidBody
from planning_through_contact.geometry.utilities import cross_2d
from planning_through_contact.tools.types import (
    NpExpressionArray,
    NpFormulaArray,
    NpVariableArray,
)


class FrictionConeDetails(NamedTuple):
    normal_vec_local: npt.NDArray[np.float64]
    R_WFc: NpExpressionArray
    p_WFc_W: NpExpressionArray
    friction_coeff: float


class StaticEquilibriumConstraints(NamedTuple):
    force_balance: NpFormulaArray
    torque_balance: NpFormulaArray

    # Rotation matrix is normalized.
    # This is to evaluate the force balance violation after the SO(2) constraints are relaxed
    normalized_force_balance: Optional[NpFormulaArray]


class ContactSceneConstraints(NamedTuple):
    pair_constraints: List[Union[LineContactConstraints, PointContactConstraints]]
    static_equilibrium_constraints: List[StaticEquilibriumConstraints]


@dataclass
class ContactSceneDefinition:
    rigid_bodies: List[RigidBody]
    contact_pairs: List[ContactPairDefinition]
    body_anchored_to_W: RigidBody

    def create_scene(
        self,
        contact_pair_modes: Dict[ContactPairDefinition, ContactMode],
        contact_pos_vars: Optional[Dict[ContactPairDefinition, sym.Variable]] = None,
        instance_postfix: Optional[str | int] = None,
    ) -> "ContactScene":
        """
        Instantiates an instance of each contact pair with new sets of variables and constraints.
        """
        if contact_pos_vars is None:
            contact_pos_vars = {}

        contact_pair_instances = [
            pair.create_pair(
                contact_pair_modes[pair],
                contact_pos_vars.get(pair, None),
                instance_postfix,
            )
            for pair in self.contact_pairs
        ]
        return ContactScene(
            self.rigid_bodies, contact_pair_instances, self.body_anchored_to_W
        )

    # TODO duplicated code
    @property
    def unactuated_bodies(self) -> List[RigidBody]:
        bodies = [body for body in self.rigid_bodies if not body.is_actuated]
        return bodies


class ContactScene:
    def __init__(
        self,
        rigid_bodies: List[RigidBody],
        contact_pairs: List[AbstractContactPair],
        body_anchored_to_W: RigidBody,
    ):
        self.rigid_bodies = rigid_bodies
        self.contact_pairs = contact_pairs
        self.body_anchored_to_W = body_anchored_to_W

    def _get_contact_forces_acting_on_body(
        self, body: RigidBody
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
        self, body: RigidBody
    ) -> List[Union[NpExpressionArray, npt.NDArray[np.float64]]]:
        contact_points = [
            pos
            for pair in self.contact_pairs
            for point in pair.contact_points
            for pos in point.get_contact_positions()
            if point.body == body
        ]
        return contact_points

    def _get_rotation_to_W(self, body: RigidBody) -> NpExpressionArray:
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

    def _get_translation_to_W(self, body: RigidBody) -> NpExpressionArray:
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
    def unactuated_bodies(self) -> List[RigidBody]:
        bodies = [body for body in self.rigid_bodies if not body.is_actuated]
        return bodies

    @property
    def variables(self) -> NpVariableArray:
        return np.concatenate([pair.variables for pair in self.contact_pairs])

    @property
    def pair_constraints(
        self,
    ) -> List[Union[LineContactConstraints, PointContactConstraints]]:
        return [pair.create_constraints() for pair in self.contact_pairs]

    def create_static_equilibrium_constraints_for_body(
        self, body: RigidBody
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
        # NOTE: The code is not unit tested, as it is not currently used. Kept around in case it will become useful.
        raise NotImplementedError("Note: This is not yet tested!")

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
        scene_def: ContactSceneDefinition,
        contact_modes: Dict[ContactPairDefinition, ContactMode],
        contact_pos_vars: Optional[Dict[ContactPairDefinition, sym.Variable]] = None,
        idx: Optional[int] = None,
    ):
        """
        Instantiates a scene instance from a scene definition.
        """

        self.contact_scene_instance = scene_def.create_scene(
            contact_modes, contact_pos_vars, idx
        )

    def get_body_rot_in_world(self, body: RigidBody) -> NpExpressionArray:
        _convert_to_expr = np.vectorize(
            lambda x: sym.Expression(x) if not isinstance(x, sym.Expression) else x
        )  # make sure we only return expressions
        R = _convert_to_expr(self.contact_scene_instance._get_rotation_to_W(body))
        return R

    def get_body_pos_in_world(self, body: RigidBody) -> NpExpressionArray:
        p_WB = self.contact_scene_instance._get_translation_to_W(body)
        _convert_to_expr = np.vectorize(
            lambda x: sym.Expression(x) if not isinstance(x, sym.Expression) else x
        )  # make sure we only return expressions
        pos_W = _convert_to_expr(p_WB)
        return pos_W

    def get_contact_forces_in_world_frame(self) -> List[NpExpressionArray]:
        forces_W = []
        for pair in self.contact_scene_instance.contact_pairs:
            for point in pair.contact_points:
                R_WB = self.contact_scene_instance._get_rotation_to_W(point.body)
                forces = point.get_contact_forces()
                f_cB_Ws = [R_WB.dot(f) for f in forces]
                forces_W.extend(f_cB_Ws)
        return forces_W

    def _get_contact_positions_for_contact_point_in_world_frame(
        self, point: ContactPoint
    ) -> List[NpExpressionArray]:
        """
        Get one contact position per contact force
        """
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
                p_Wc1_Ws = self._get_contact_positions_for_contact_point_in_world_frame(
                    point
                )
                pos_W.extend(p_Wc1_Ws)
        return pos_W

    # NOTE: Friction cone functionality (used only for visualization) is commented out as
    # it does not work. There is some logic error with which rotation and position is used.
    # Kept around in case I want to fix it.

    # def _get_friction_cone_details(
    #     self, point: ContactPoint, force: ContactForce
    # ) -> List[FrictionConeDetails]:
    #     normal_vec = force.normal_vec
    #     if normal_vec is None:
    #         raise ValueError("Could not get normal vector for force")
    #     R_WFc = self.contact_scene_instance._get_rotation_to_W(point.body)
    #     p_WFc_Ws = self._get_contact_positions_for_contact_point_in_world_frame(point)
    #     friction_cone_details = [
    #         FrictionConeDetails(normal_vec, R_WFc, p_WFc_W, force.friction_coeff)
    #         for p_WFc_W in p_WFc_Ws
    #     ]
    #     return friction_cone_details
    #
    # def get_friction_cones_details_for_face_contact_points(
    #     self,
    # ) -> List[FrictionConeDetails]:
    #     contact_points_on_faces = [
    #         pair.get_nonfixed_contact_point()
    #         for pair in self.contact_scene_instance.contact_pairs
    #     ]
    #     return [
    #         details
    #         for point in contact_points_on_faces
    #         for force in point.contact_forces
    #         for details in self._get_friction_cone_details(point, force)
    #         if force.normal_vec is not None
    #     ]

    @property
    def static_equilibrium_constraints(self) -> List[StaticEquilibriumConstraints]:
        return self.constraints.static_equilibrium_constraints

    @property
    def friction_cone_constraints(self) -> NpFormulaArray:
        return np.concatenate(
            [c.friction_cone for c in self.constraints.pair_constraints]
        ).flatten()

    @property
    def so_2_constraints(self) -> NpFormulaArray:
        return np.array(
            [c.so_2 for c in self.point_on_line_contact_constraints]
        ).flatten()

    @property
    def rotation_bounds(self) -> NpFormulaArray:
        return np.array(
            [c.rotation_bounds for c in self.point_on_line_contact_constraints]
        ).flatten()

    @property
    def convex_hull_bounds(self) -> NpFormulaArray:
        return np.array(
            [c.convex_hull_bounds for c in self.constraints.pair_constraints]
        ).flatten()

    @property
    def relaxed_so_2_constraints(self) -> NpFormulaArray:
        return np.array(
            [c.relaxed_so_2 for c in self.point_on_line_contact_constraints]
        )

    @property
    def non_penetration_cuts(self) -> NpFormulaArray:
        for c in self.point_on_line_contact_constraints:
            if c.non_penetration_cut is None:
                raise NotImplementedError(
                    "Cannot get non-penetration cut: value is None"
                )

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
    def point_on_line_contact_constraints(self) -> List[PointContactConstraints]:
        return [
            constraints
            for constraints in self.constraints.pair_constraints
            if isinstance(constraints, PointContactConstraints)
        ]
