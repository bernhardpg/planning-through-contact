from typing import Any, Dict, List, Literal, Tuple, Union

import numpy as np
import numpy.typing as npt
from pydrake.math import eq, ge, le
from pydrake.solvers import MathematicalProgram, Solve

from convex_relaxation.mccormick import (
    add_bilinear_constraints_to_prog,
    add_bilinear_frame_constraints_to_prog,
)
from geometry.two_d.contact.contact_pair_2d import ContactPair2d
from geometry.two_d.contact.contact_scene_2d import (
    ContactScene2d,
    ContactSceneCtrlPoint,
)
from geometry.two_d.contact.types import ContactMode
from geometry.utilities import two_d_rotation_matrix_from_angle
from tools.types import NpExpressionArray, NpVariableArray


class ContactModeMotionPlanner:
    def __init__(
        self,
        contact_scene: ContactScene2d,
        num_ctrl_points: int,
        contact_modes: Dict[str, ContactMode],
        variable_bounds: Dict[str, Tuple[float, float]],
    ):

        # Convenience variables for running experiments
        self.use_friction_cone_constraint = True
        self.use_force_balance_constraint = True
        self.use_torque_balance_constraint = True
        self.use_equal_contact_point_constraint = True
        self.use_equal_relative_position_constraint = False  # Not in use as it does not make the solution any tighter without variable bounds
        self.use_equal_and_opposite_forces_constraint = True
        self.use_so2_constraint = False
        self.use_non_penetration_cut = True
        self.use_quadratic_cost = True
        self.only_minimize_forces_on_unactuated_bodies = True

        self.contact_modes = contact_modes
        self.contact_scene = contact_scene
        self.num_ctrl_points = num_ctrl_points
        self._setup_ctrl_points()
        self._setup_prog(variable_bounds)

    def _setup_ctrl_points(self) -> None:
        self.ctrl_points = [
            ContactSceneCtrlPoint(
                self.contact_scene.create_instance(self.contact_modes)
            )
            for _ in range(self.num_ctrl_points)
        ]

    def _setup_prog(self, variable_bounds: Dict[str, Tuple[float, float]]) -> None:
        self.prog = MathematicalProgram()

        for ctrl_point in self.ctrl_points:
            self.prog.AddDecisionVariables(ctrl_point.variables)

            if self.use_friction_cone_constraint:
                self.prog.AddLinearConstraint(ctrl_point.friction_cone_constraints)

            if self.use_force_balance_constraint:
                for c in ctrl_point.static_equilibrium_constraints:
                    self.prog.AddLinearConstraint(c.force_balance)

            if self.use_torque_balance_constraint:
                for c in ctrl_point.static_equilibrium_constraints:
                    add_bilinear_constraints_to_prog(
                        c.torque_balance,
                        self.prog,
                        variable_bounds,
                    )

            if self.use_equal_contact_point_constraint:
                for c in ctrl_point.equal_contact_point_constraints:
                    add_bilinear_frame_constraints_to_prog(
                        c, self.prog, variable_bounds
                    )

            if self.use_equal_relative_position_constraint:
                for c in ctrl_point.equal_rel_position_constraints:
                    add_bilinear_frame_constraints_to_prog(
                        c, self.prog, variable_bounds
                    )

            if self.use_equal_and_opposite_forces_constraint:
                for c in ctrl_point.equal_and_opposite_forces_constraints:
                    add_bilinear_frame_constraints_to_prog(
                        c, self.prog, variable_bounds
                    )

            if self.use_so2_constraint:
                for c in ctrl_point.relaxed_so_2_constraints:
                    lhs, rhs = c.Unapply()[1]
                    self.prog.AddLorentzConeConstraint(rhs, lhs)  # type: ignore

            if self.use_non_penetration_cut:
                self.prog.AddLinearConstraint(ctrl_point.non_penetration_cuts)

            if self.use_quadratic_cost:
                self.prog.AddQuadraticCost(
                    ctrl_point.get_squared_forces(
                        self.only_minimize_forces_on_unactuated_bodies
                    )
                )
            else:  # Absolute value cost
                raise ValueError("Absolute value cost not implemented")

        for pair, mode in self.contact_modes.items():
            if mode == ContactMode.ROLLING:
                self._fix_contact_positions(pair)
            elif mode == ContactMode.SLIDING_LEFT:
                self._constrain_contact_velocity(pair, "NEGATIVE")
            elif mode == ContactMode.SLIDING_RIGHT:
                self._constrain_contact_velocity(pair, "POSITIVE")

    def _get_contact_pos_for_pair(self, pair_name) -> List[NpExpressionArray]:
        pair = next(
            pair for pair in self.contact_scene.contact_pairs if pair.name == pair_name
        )
        pair_at_ctrl_points = [
            next(
                pair_instance
                for pair_instance in ctrl_point.contact_scene_instance.contact_pairs
                if pair_instance.name == pair.name
            )
            for ctrl_point in self.ctrl_points
        ]
        contact_pos_at_ctrl_points = [
            pair.get_nonfixed_contact_position() for pair in pair_at_ctrl_points
        ]  # [(num_dims, 1) x num_ctrl_points]
        return contact_pos_at_ctrl_points

    def _fix_contact_positions(self, pair_name: str) -> None:
        contact_pos_at_ctrl_points = self._get_contact_pos_for_pair(pair_name)
        for idx in range(self.num_ctrl_points - 1):
            constraint = eq(
                contact_pos_at_ctrl_points[idx], contact_pos_at_ctrl_points[idx + 1]
            )
            for c in constraint.flatten():
                self.prog.AddLinearConstraint(c)

    def _constrain_contact_velocity(
        self, pair_name: str, direction: Literal["POSITIVE", "NEGATIVE"]
    ) -> None:
        contact_pos_at_ctrl_points = self._get_contact_pos_for_pair(pair_name)
        for idx in range(self.num_ctrl_points - 1):
            contact_velocity = (
                contact_pos_at_ctrl_points[idx + 1] - contact_pos_at_ctrl_points[idx]
            )  # type: ignore
            if direction == "POSITIVE":
                constraint = ge(contact_velocity, 0)
            elif direction == "NEGATIVE":
                constraint = le(contact_velocity, 0)
            else:
                raise ValueError("Direction must be either positive or negative")

            for c in constraint.flatten():
                self.prog.AddLinearConstraint(c)

    def constrain_orientation_at_ctrl_point(
        self, pair_to_constrain: ContactPair2d, ctrl_point_idx: int, theta: float
    ) -> None:
        # NOTE: This finds the matching pair based on name. This may not be the safest way to do this

        scene_instance = self.ctrl_points[ctrl_point_idx].contact_scene_instance
        pair = next(
            p for p in scene_instance.contact_pairs if p.name == pair_to_constrain.name
        )
        R_target = two_d_rotation_matrix_from_angle(theta)
        constraint = eq(pair.R_AB, R_target)

        for c in constraint.flatten():
            self.prog.AddLinearConstraint(c)

    def constrain_contact_position_at_ctrl_point(
        self, pair_to_constrain: ContactPair2d, ctrl_point_idx: int, lam_target: float
    ) -> None:
        """
        Constraints position by fixing position along contact face. lam_target should take values in the range [0,1]
        """
        if lam_target > 1.0 or lam_target < 0.0:
            raise ValueError("lam_target must be in the range [0, 1]")

        # NOTE: This finds the matching pair based on name. This may not be the safest way to do this
        scene_instance = self.ctrl_points[ctrl_point_idx].contact_scene_instance
        pair = next(
            p for p in scene_instance.contact_pairs if p.name == pair_to_constrain.name
        )
        constraint = pair.get_nonfixed_contact_point_variable() == lam_target
        self.prog.AddLinearConstraint(constraint)

    def solve(self) -> None:
        self.result = Solve(self.prog)
        print(f"Solution result: {self.result.get_solution_result()}")
        assert self.result.is_success()

        print(f"Cost: {self.result.get_optimal_cost()}")

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
        num_elements = len(contact_orientations_for_each_ctrl_point[0])
        ctrl_points_per_element = [
            [elements[idx] for elements in contact_orientations_for_each_ctrl_point]
            for idx in range(num_elements)
        ]  # [[(2,2) x num_ctrl_points] x num_bodies]

        return ctrl_points_per_element

    @property
    def body_orientations(
        self,
    ) -> List[List[Union[NpExpressionArray, NpVariableArray]]]:
        body_orientations_for_each_ctrl_point = [
            cp.get_body_orientations() for cp in self.ctrl_points
        ]
        num_elements = len(body_orientations_for_each_ctrl_point[0])
        ctrl_points_per_element = [
            [elements[idx] for elements in body_orientations_for_each_ctrl_point]
            for idx in range(num_elements)
        ]  # [[(2,2) x num_ctrl_points] x num_bodies]

        return ctrl_points_per_element

    @property
    def contact_point_normals_in_local_frames(self) -> List[npt.NDArray[np.float64]]:
        return self.ctrl_points[
            0  # contact normals are constant, so we just pick them from the first ctrl point
        ].contact_scene_instance.get_contact_normals_in_local_frames()
