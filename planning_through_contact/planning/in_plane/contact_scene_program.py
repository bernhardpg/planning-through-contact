from typing import Any, Dict, List, Literal, Optional, Tuple, TypeVar, Union

import numpy as np
import numpy.typing as npt
from pydrake.math import eq, ge, le
from pydrake.solvers import MathematicalProgram, Solve
from pydrake.symbolic import Variable

from planning_through_contact.convex_relaxation.mccormick import (
    add_bilinear_constraints_to_prog,
    add_bilinear_frame_constraints_to_prog,
)
from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    ContactMode,
)
from planning_through_contact.geometry.in_plane.contact_pair import (
    ContactFrameConstraints,
    ContactPairDefinition,
)
from planning_through_contact.geometry.in_plane.contact_scene import (
    ContactSceneCtrlPoint,
    ContactSceneDefinition,
)
from planning_through_contact.geometry.rigid_body import RigidBody
from planning_through_contact.geometry.utilities import two_d_rotation_matrix_from_angle
from planning_through_contact.tools.types import NpExpressionArray, NpVariableArray
from planning_through_contact.tools.utils import convert_formula_to_lhs_expression


class ContactSceneProgram:
    def __init__(
        self,
        contact_scene_def: ContactSceneDefinition,
        num_ctrl_points: int,
        contact_modes: Dict[ContactPairDefinition, ContactMode],
    ):
        # Convenience variables for running experiments
        self.use_friction_cone_constraint = True
        # TODO turn back on the constraints I need
        self.use_force_balance_constraint = True
        self.use_torque_balance_constraint = True
        self.use_equal_contact_point_constraint = True
        self.use_equal_relative_position_constraint = True  # Not in use as it does not make the solution any tighter without variable bounds
        self.use_equal_and_opposite_forces_constraint = True
        self.use_so2_constraint = True
        self.minimize_velocities = True

        self.contact_modes = contact_modes
        self.contact_scene_def = contact_scene_def
        self.num_ctrl_points = num_ctrl_points
        self._setup_ctrl_points()
        self._setup_prog()

    def _setup_ctrl_points(self) -> None:
        # Create contact position variables to be shared between knot points for rolling contacts
        self.contact_pos_vars = {
            pair: Variable(f"{pair.name}_lam")
            for pair, mode in self.contact_modes.items()
            if mode == ContactMode.ROLLING
        }

        self.ctrl_points = [
            ContactSceneCtrlPoint(
                self.contact_scene_def, self.contact_modes, self.contact_pos_vars, idx
            )
            for idx in range(self.num_ctrl_points)
        ]

    def _add_frame_constraints_to_prog(self, c: ContactFrameConstraints) -> None:
        """
        Properly adds the constraints to MathematicalProgram such that they are processed
        as either quadratic or linear equalities (and not 'generic')
        """
        if c.type_A == "linear":
            for lin_c in c.in_frame_A:
                self.prog.AddLinearEqualityConstraint(lin_c)
        else:  # quadratic
            for quad_c in c.in_frame_A:
                self.prog.AddQuadraticConstraint(
                    convert_formula_to_lhs_expression(quad_c), 0, 0
                )
        if c.type_B == "linear":
            for lin_c in c.in_frame_B:
                self.prog.AddLinearEqualityConstraint(lin_c)
        else:  # quadratic
            for quad_c in c.in_frame_B:
                self.prog.AddQuadraticConstraint(
                    convert_formula_to_lhs_expression(quad_c), 0, 0
                )

    def _setup_prog(self) -> None:
        self.prog = MathematicalProgram()

        for ctrl_point in self.ctrl_points:
            self.prog.AddDecisionVariables(ctrl_point.variables)

            self.prog.AddLinearConstraint(ctrl_point.convex_hull_bounds)

            if self.use_friction_cone_constraint:
                self.prog.AddLinearConstraint(ctrl_point.friction_cone_constraints)

            if self.use_so2_constraint:
                for quadratic_const in ctrl_point.so_2_constraints:
                    expr = convert_formula_to_lhs_expression(quadratic_const)
                    self.prog.AddQuadraticConstraint(expr, 0, 0)

                # Bounding box constraints
                self.prog.AddLinearConstraint(ctrl_point.rotation_bounds)

            for (
                force_balance,
                torque_balance,
                _,
            ) in ctrl_point.static_equilibrium_constraints:
                torque_balance = torque_balance.item()
                if self.use_torque_balance_constraint:
                    expr = convert_formula_to_lhs_expression(torque_balance)
                    self.prog.AddQuadraticConstraint(expr, 0, 0)

                if self.use_force_balance_constraint:
                    self.prog.AddLinearConstraint(force_balance)

            if self.use_equal_contact_point_constraint:
                for c in ctrl_point.equal_contact_point_constraints:
                    self._add_frame_constraints_to_prog(c)

            if self.use_equal_relative_position_constraint:
                for c in ctrl_point.equal_rel_position_constraints:
                    self._add_frame_constraints_to_prog(c)

            if self.use_equal_and_opposite_forces_constraint:
                for c in ctrl_point.equal_and_opposite_forces_constraints:
                    self._add_frame_constraints_to_prog(c)

        for pair, mode in self.contact_modes.items():
            if mode == ContactMode.ROLLING:
                pass  # we are already using one shared contact position variable for all rolling contacts
            elif mode == ContactMode.SLIDING_LEFT:
                self._constrain_contact_velocity(pair, "NEGATIVE")
            elif mode == ContactMode.SLIDING_RIGHT:
                self._constrain_contact_velocity(pair, "POSITIVE")

        if self.minimize_velocities:
            if len(self.contact_scene_def.unactuated_bodies) > 1:
                raise NotImplementedError(
                    "Currently only support for one unactuated body."
                )

            unactuated_body = self.contact_scene_def.unactuated_bodies[0]
            r_dot_sq = self._get_sq_rot_param_dots(unactuated_body)
            for c in r_dot_sq:
                self.prog.AddQuadraticCost(c)  # type: ignore

            pos_dot_sq = self._get_sq_body_vels(unactuated_body)
            for c in pos_dot_sq:
                self.prog.AddQuadraticCost(c)  # type: ignore

    def _get_sq_rot_param_dots(self, body: RigidBody) -> List[NpExpressionArray]:
        rot_params = [
            ctrl_point.contact_scene_instance._get_rotation_to_W(body).flatten()[
                0:2
            ]  # only keep one copy of sin and cos
            for ctrl_point in self.ctrl_points
        ]
        rot_params_dot = self._get_vel_from_pos_by_fe(rot_params, dt=None)
        r_dot_sq = [r_dot.T.dot(r_dot) for r_dot in rot_params_dot]

        return r_dot_sq

    def _get_sq_body_vels(self, body: RigidBody) -> List[NpExpressionArray]:
        pos = [
            ctrl_point.contact_scene_instance._get_translation_to_W(body).flatten()
            for ctrl_point in self.ctrl_points
        ]
        vels = self._get_vel_from_pos_by_fe(pos, dt=None)
        vel_sq = [vel.T.dot(vel) for vel in vels]

        return vel_sq

    def constrain_orientation_at_ctrl_point(
        self,
        pair_to_constrain: ContactPairDefinition,
        ctrl_point_idx: int,
        theta: float,
    ) -> None:
        scene_instance = self.ctrl_points[ctrl_point_idx].contact_scene_instance
        pair = next(
            p for p in scene_instance.contact_pairs if p.definition == pair_to_constrain
        )
        R_target = two_d_rotation_matrix_from_angle(theta)
        constraint = eq(
            pair.R_AB[:, 0], R_target[:, 0]
        ).flatten()  # remove extra column

        self.prog.AddLinearConstraint(constraint)

    def constrain_contact_position_at_ctrl_point(
        self,
        pair_to_constrain: ContactPairDefinition,
        ctrl_point_idx: int,
        lam_target: float,
    ) -> None:
        """
        Constraints position by fixing position along contact face. lam_target should take values in the range [0,1]
        """
        if lam_target > 1.0 or lam_target < 0.0:
            raise ValueError("lam_target must be in the range [0, 1]")

        scene_instance = self.ctrl_points[ctrl_point_idx].contact_scene_instance
        pair = next(
            p for p in scene_instance.contact_pairs if p.definition == pair_to_constrain
        )
        expr = pair.get_nonfixed_contact_point_variable() - lam_target
        self.prog.AddLinearEqualityConstraint(expr == 0)

    T = TypeVar("T", bound=Any)

    def _get_vel_from_pos_by_fe(
        self, pos: List[npt.NDArray[T]], dt: Optional[float] = None
    ) -> List[npt.NDArray[T]]:
        """
        Returns a forward euler approximation to the velocity

        x_dot[i] = (x[i+1] - x[i]) / dt

        @param pos: List of positions for each point in time
        @param dt: If dt is none, the velocities will be unscaled.
        """
        if dt is None:
            dt = 1.0  # no scaling

        vels = [(pos[idx + 1] - pos[idx]) / dt for idx in range(len(pos) - 1)]
        return vels  # type: ignore

    def _get_nonfixed_contact_pos_for_pair(
        self, pair: ContactPairDefinition
    ) -> List[NpExpressionArray]:
        pair_at_ctrl_points = [
            next(
                pair_instance
                for pair_instance in ctrl_point.contact_scene_instance.contact_pairs
                if pair_instance.definition == pair
            )
            for ctrl_point in self.ctrl_points
        ]
        contact_pos_at_ctrl_points = [
            pair.get_nonfixed_contact_position() for pair in pair_at_ctrl_points
        ]  # [(num_dims, 1) x num_ctrl_points]

        return contact_pos_at_ctrl_points  # type: ignore

    def _constrain_contact_velocity(
        self, pair: ContactPairDefinition, direction: Literal["POSITIVE", "NEGATIVE"]
    ) -> None:
        contact_pos_per_ctrl_point = self._get_nonfixed_contact_pos_for_pair(pair)
        unscaled_contact_vels = self._get_vel_from_pos_by_fe(
            contact_pos_per_ctrl_point, dt=None
        )

        for vel in unscaled_contact_vels:
            if direction == "POSITIVE":
                constraint = ge(vel, 0).flatten()
            else:  # "NEGATIVE"
                constraint = le(vel, 0).flatten()

            self.prog.AddLinearConstraint(constraint)

    # TODO: Remove all of these functions from here and down:
    ##########

    @staticmethod
    def _collect_ctrl_points(
        elements_per_ctrl_point: List[List[npt.NDArray[Any]]],
    ) -> List[npt.NDArray[Any]]:
        """
        Helper function for creating ctrl points.

        Input: [[(num_dims, 1) x num_elements] x num_ctrl_points]
        Output: [(num_dims, num_ctrl_points) x num_elements]

        Takes in a list of lists with vectors. The outermost list has one entry per ctrl point,
        the inner list has an entry per element.
        The function returns a list of collected ctrl points per element.
        """
        num_elements = len(elements_per_ctrl_point[0])
        ctrl_points_per_element = [
            np.hstack([elements[idx] for elements in elements_per_ctrl_point])
            for idx in range(num_elements)
        ]  # [(num_dims, num_ctrl_points) x num_forces]
        return ctrl_points_per_element

    @property
    def contact_forces_in_world_frame(self) -> List[NpExpressionArray]:
        contact_forces_for_each_ctrl_point = [
            cp.get_contact_forces_in_world_frame() for cp in self.ctrl_points
        ]
        return self._collect_ctrl_points(contact_forces_for_each_ctrl_point)

    @property
    def contact_positions_in_world_frame(self) -> List[NpExpressionArray]:
        contact_positions_for_each_ctrl_point = [
            cp.get_contact_positions_in_world_frame() for cp in self.ctrl_points
        ]
        return self._collect_ctrl_points(contact_positions_for_each_ctrl_point)

    @property
    def gravitational_forces_in_world_frame(self) -> List[npt.NDArray[np.float64]]:
        gravity_force_for_each_ctrl_point = [
            cp.get_gravitational_forces_in_world_frame() for cp in self.ctrl_points
        ]
        return self._collect_ctrl_points(gravity_force_for_each_ctrl_point)

    @property
    def body_positions_in_world_frame(
        self,
    ) -> List[Union[NpExpressionArray, npt.NDArray[np.float64]]]:
        body_positions_for_each_ctrl_point = [
            cp.get_body_positions_in_world_frame() for cp in self.ctrl_points
        ]
        return self._collect_ctrl_points(body_positions_for_each_ctrl_point)

    @staticmethod
    def _sort_by_elements(
        sorted_by_ctrl_points: List[List[NpExpressionArray]],
    ) -> List[List[NpExpressionArray]]:
        num_elements = len(sorted_by_ctrl_points[0])
        sorted_per_element = [
            [elements[idx] for elements in sorted_by_ctrl_points]
            for idx in range(num_elements)
        ]  # [[(N,M) x num_ctrl_points] x num_elements]
        return sorted_per_element

    @property
    def contact_point_orientations(
        self,
    ) -> List[List[Union[NpExpressionArray, NpVariableArray]]]:
        """
        Exactly the same as body_orientations, but returns one orientation corresponding to each contact point.
        Useful when plotting friction cones
        """
        contact_orientations_for_each_ctrl_point = [
            cp.get_contact_point_orientations() for cp in self.ctrl_points
        ]
        return self._sort_by_elements(contact_orientations_for_each_ctrl_point)

    @property
    def body_orientations(
        self,
    ) -> List[List[Union[NpExpressionArray, NpVariableArray]]]:
        body_orientations_for_each_ctrl_point = [
            cp.get_body_orientations() for cp in self.ctrl_points
        ]
        return self._sort_by_elements(body_orientations_for_each_ctrl_point)

    @property
    def contact_point_friction_cones(
        self,
    ) -> Tuple[
        List[npt.NDArray[np.float64]],
        List[NpExpressionArray],
        List[List[NpExpressionArray]],
    ]:
        friction_cone_details_per_ctrl_point = [
            cp.get_friction_cones_details_for_face_contact_points()
            for cp in self.ctrl_points
        ]

        normals = self._collect_ctrl_points(
            [
                [fc.normal_vec_local for fc in cp]
                for cp in friction_cone_details_per_ctrl_point
            ]
        )
        positions = self._collect_ctrl_points(
            [[fc.p_WFc_W for fc in cp] for cp in friction_cone_details_per_ctrl_point]
        )

        orientations = self._sort_by_elements(
            [[fc.R_WFc for fc in cp] for cp in friction_cone_details_per_ctrl_point]
        )

        return normals, positions, orientations
