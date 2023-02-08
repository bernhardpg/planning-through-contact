import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, TypeVar, Union

import numpy as np
import numpy.typing as npt
import pydrake.symbolic as sym  # type: ignore
from pydrake.math import eq
from pydrake.solvers import MathematicalProgram, SolutionResult, Solve

from convex_relaxation.mccormick import (
    add_bilinear_constraints_to_prog,
    add_bilinear_frame_constraints_to_prog,
)
from geometry.two_d.box_2d import Box2d, RigidBody2d
from geometry.two_d.contact.contact_pair_2d import (
    ContactFrameConstraints,
    ContactPair2d,
    EvaluatedContactFrameConstraints,
)
from geometry.two_d.contact.contact_scene_2d import (
    ContactScene2d,
    ContactSceneCtrlPoint,
    ContactSceneInstance,
)
from geometry.two_d.contact.types import ContactMode, ContactPosition, ContactType
from geometry.two_d.equilateral_polytope_2d import EquilateralPolytope2d
from geometry.two_d.rigid_body_2d import PolytopeContactLocation
from geometry.utilities import two_d_rotation_matrix_from_angle
from tools.types import NpExpressionArray, NpFormulaArray, NpVariableArray
from tools.utils import evaluate_np_expressions_array, evaluate_np_formulas_array
from visualize.analysis import (
    create_force_plot,
    create_newtons_third_law_analysis,
    create_static_equilibrium_analysis,
    show_plots,
)
from visualize.visualizer_2d import (
    VisualizationForce2d,
    VisualizationPoint2d,
    VisualizationPolygon2d,
    Visualizer2d,
)

# FIX: Only defined here because of poor variable bound code. Should be removed
FRICTION_COEFF = 0.7


class ContactMotionPlan:
    def __init__(
        self,
        contact_scene: ContactScene2d,
        num_ctrl_points: int,
        variable_bounds: Dict[str, Tuple[float, float]],
    ):

        # Convenience variables for running experiments
        self.use_friction_cone_constraint = True
        self.use_force_balance_constraint = True
        self.use_torque_balance_constraint = True
        self.use_equal_contact_point_constraint = True
        self.use_equal_relative_position_constraint = False  # Not in use as it does not make the solution any tighter without variable bounds
        self.use_equal_and_opposite_forces_constraint = True
        self.use_so2_constraint = True
        self.use_non_penetration_cut = True
        self.use_quadratic_cost = True

        self.contact_scene = contact_scene
        self.num_ctrl_points = num_ctrl_points
        self._setup_ctrl_points()
        self._setup_prog(variable_bounds)

    def _setup_ctrl_points(self) -> None:
        modes = {"contact_1": ContactMode.ROLLING, "contact_2": ContactMode.ROLLING}
        self.ctrl_points = [
            ContactSceneCtrlPoint(self.contact_scene.create_instance(modes))
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
                self.prog.AddQuadraticCost(ctrl_point.squared_forces)
            else:  # Absolute value cost
                raise ValueError("Absolute value cost not implemented")

    def fix_contact_positions(self) -> None:
        for pair in self.contact_scene.contact_pairs:
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
            ]

            for idx in range(self.num_ctrl_points - 1):
                constraint = eq(
                    contact_pos_at_ctrl_points[idx], contact_pos_at_ctrl_points[idx + 1]
                )
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


def plan_triangle_flipup():
    TABLE_HEIGHT = 0.5
    TABLE_WIDTH = 2

    FINGER_HEIGHT = 0.1
    FINGER_WIDTH = 0.1

    TRIANGLE_MASS = 1

    triangle = EquilateralPolytope2d(
        actuated=False,
        name="triangle",
        mass=TRIANGLE_MASS,
        vertex_distance=0.2,
        num_vertices=3,
    )
    table = Box2d(
        actuated=True,
        name="table",
        mass=None,
        width=TABLE_WIDTH,
        height=TABLE_HEIGHT,
    )
    finger = Box2d(
        actuated=True,
        name="finger",
        mass=None,
        width=FINGER_WIDTH,
        height=FINGER_HEIGHT,
    )
    table_triangle = ContactPair2d(
        "contact_1",
        table,
        PolytopeContactLocation(ContactPosition.FACE, 1),
        triangle,
        PolytopeContactLocation(ContactPosition.VERTEX, 2),
        ContactType.POINT_CONTACT,
        FRICTION_COEFF,
    )
    triangle_finger = ContactPair2d(
        "contact_2",
        triangle,
        PolytopeContactLocation(ContactPosition.FACE, 0),
        finger,
        PolytopeContactLocation(ContactPosition.VERTEX, 1),
        ContactType.POINT_CONTACT,
        FRICTION_COEFF,
    )
    contact_scene = ContactScene2d(
        [table, triangle, finger],
        [table_triangle, triangle_finger],
        table,
    )

    # TODO: this should be cleaned up
    MAX_FORCE = TRIANGLE_MASS * 9.81 * 2  # only used for mccorimick constraints
    variable_bounds = {
        "contact_1_triangle_c_n": (0.0, MAX_FORCE),
        "contact_1_triangle_c_f": (
            -FRICTION_COEFF * MAX_FORCE,
            FRICTION_COEFF * MAX_FORCE,
        ),
        "contact_1_table_c_n": (0.0, MAX_FORCE),
        "contact_1_table_c_f": (
            -FRICTION_COEFF * MAX_FORCE,
            FRICTION_COEFF * MAX_FORCE,
        ),
        "contact_1_table_lam": (0.0, 1.0),
        "contact_1_sin_th": (-1, 1),
        "contact_1_cos_th": (-1, 1),
        "contact_2_triangle_lam": (0.0, 1.0),
        "contact_2_triangle_c_n": (0, MAX_FORCE / 2),
        "contact_2_triangle_c_f": (0, MAX_FORCE / 2),
        "contact_2_sin_th": (-1, 1),
        "contact_2_cos_th": (-1, 1),
    }

    num_ctrl_points = 3
    motion_plan = ContactMotionPlan(contact_scene, num_ctrl_points, variable_bounds)
    motion_plan.constrain_orientation_at_ctrl_point(
        table_triangle, ctrl_point_idx=0, theta=0
    )
    motion_plan.constrain_orientation_at_ctrl_point(
        table_triangle, ctrl_point_idx=num_ctrl_points - 1, theta=np.pi / 4
    )
    motion_plan.fix_contact_positions()
    motion_plan.solve()

    contact_positions_ctrl_points = [
        evaluate_np_expressions_array(pos, motion_plan.result)
        for pos in motion_plan.contact_positions_in_world_frame
    ]
    contact_forces_ctrl_points = [
        evaluate_np_expressions_array(force, motion_plan.result)
        for force in motion_plan.contact_forces_in_world_frame
    ]

    if True:

        CONTACT_COLOR = "brown1"
        GRAVITY_COLOR = "blueviolet"
        BOX_COLOR = "aquamarine4"
        TABLE_COLOR = "bisque3"
        body_colors = [TABLE_COLOR, BOX_COLOR]

        viz_contact_positions = [
            VisualizationPoint2d.from_ctrl_points(pos, CONTACT_COLOR)
            for pos in contact_positions_ctrl_points
        ]
        viz_contact_forces = [
            VisualizationForce2d.from_ctrl_points(pos, force, CONTACT_COLOR)
            for pos, force in zip(
                contact_positions_ctrl_points, contact_forces_ctrl_points
            )
        ]

        bodies_com_ctrl_points = [
            motion_plan.result.GetSolution(ctrl_point)
            for ctrl_point in motion_plan.body_positions_in_world_frame
        ]

        viz_com_points = [
            VisualizationPoint2d.from_ctrl_points(com_ctrl_points, GRAVITY_COLOR)
            for com_ctrl_points in bodies_com_ctrl_points
        ]

        box_com_ctrl_points = bodies_com_ctrl_points[1]
        # TODO: should not depend explicitly on box
        viz_gravitional_forces = [
            VisualizationForce2d.from_ctrl_points(
                box_com_ctrl_points,
                motion_plan.result.GetSolution(force_ctrl_points),
                GRAVITY_COLOR,
            )
            for force_ctrl_points in motion_plan.gravitational_forces_in_world_frame
        ]

        bodies_orientation_ctrl_points = [
            [motion_plan.result.GetSolution(R_ctrl_point) for R_ctrl_point in R]
            for R in motion_plan.body_orientations
        ]
        viz_polygons = [
            VisualizationPolygon2d.from_ctrl_points(
                com,
                orientation,
                body,
                color,
            )
            for com, orientation, body, color in zip(
                bodies_com_ctrl_points,
                bodies_orientation_ctrl_points,
                contact_scene.rigid_bodies,
                body_colors,
            )
        ]

        viz = Visualizer2d()
        viz.visualize(
            viz_contact_positions + viz_com_points,
            viz_contact_forces + viz_gravitional_forces,
            viz_polygons,
        )
    breakpoint()


if __name__ == "__main__":
    plan_triangle_flipup()
