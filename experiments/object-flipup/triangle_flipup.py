from typing import Dict, Tuple

import numpy as np
import pydrake.symbolic as sym
from pydrake.math import eq
from pydrake.solvers import MathematicalProgram, SolutionResult, Solve

from convex_relaxation.mccormick import (
    add_bilinear_constraints_to_prog,
    add_bilinear_frame_constraints_to_prog,
)
from geometry.two_d.box_2d import Box2d
from geometry.two_d.contact.contact_pair_2d import ContactPair2d
from geometry.two_d.contact.contact_scene_2d import (
    ContactScene2d,
    ContactSceneCtrlPoint,
)
from geometry.two_d.contact.types import ContactMode, ContactPosition, ContactType
from geometry.two_d.equilateral_polytope_2d import EquilateralPolytope2d
from geometry.two_d.rigid_body_2d import PolytopeContactLocation
from geometry.utilities import two_d_rotation_matrix_from_angle

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
        self.use_equal_contact_point_constraint = (
            False  # Not in use as it does not add any tightness without variable bounds
        )
        self.use_equal_relative_position_constraint = True
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

        # # Don't allow contact position to change
        # for idx in range(self.num_ctrl_points - 1):
        #     self.prog.AddLinearConstraint(
        #         eq(self.ctrl_points[idx].pc2_B, self.ctrl_points[idx + 1].pc2_B)
        #     )
        #     self.prog.AddLinearConstraint(
        #         eq(self.ctrl_points[idx].pc1_T, self.ctrl_points[idx + 1].pc1_T)
        #     )

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
        if self.result.get_solution_result() == SolutionResult.kUnknownError:
            print(
                "NOTE! Got kUnknownError from Mosek, which most likely means numerical issues"
            )
            pass
        else:  # NOTE: We do not care if the solution is kUnknownError as we already know that this will happen in some cases
            assert self.result.is_success()

        print(f"Cost: {self.result.get_optimal_cost()}")


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
    MAX_FORCE = 10  # only used for mccorimick constraints
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
        "contact_2_triangle_c_n": (0, 3.8),
    }

    num_ctrl_points = 3
    motion_plan = ContactMotionPlan(contact_scene, num_ctrl_points, variable_bounds)
    motion_plan.constrain_orientation_at_ctrl_point(
        table_triangle, ctrl_point_idx=0, theta=0
    )
    motion_plan.constrain_orientation_at_ctrl_point(
        table_triangle, ctrl_point_idx=num_ctrl_points - 1, theta=np.pi / 4
    )
    motion_plan.solve()
    breakpoint()


if __name__ == "__main__":
    plan_triangle_flipup()
