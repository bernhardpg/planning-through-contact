import argparse
from dataclasses import dataclass
from typing import List, TypeVar, Union

import numpy as np
import numpy.typing as npt
import pydrake.symbolic as sym  # type: ignore
from pydrake.math import eq
from pydrake.solvers import MathematicalProgram, Solve

from convex_relaxation.mccormick import (
    add_bilinear_constraints_to_prog,
    add_bilinear_frame_constraints_to_prog,
)
from geometry.hyperplane import construct_2d_plane_from_points
from geometry.two_d.box_2d import Box2d
from geometry.two_d.contact.contact_pair_2d import (
    ContactPair2d,
    EvaluatedContactFrameConstraints,
)
from geometry.two_d.contact.contact_scene_2d import ContactScene2d
from geometry.two_d.contact.types import ContactLocation, ContactMode, ContactType
from geometry.two_d.rigid_body_2d import Point2d
from geometry.utilities import cross_2d
from tools.types import NpExpressionArray, NpVariableArray
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

# FIX: should not be defined here
NUM_CTRL_POINTS = 3

T = TypeVar("T")


def _angle_to_2d_rot_matrix(theta: float) -> npt.NDArray[np.float64]:
    rot_matrix = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    return rot_matrix


@dataclass
class BoxFlipupCtrlPoint:
    table: Box2d
    box: Box2d
    finger: Point2d
    friction_coeff: float
    idx: int

    def _create_contact_pairs(self) -> None:
        self.table_box = ContactPair2d(
            "contact_1",
            self.table,
            ContactLocation(ContactType.FACE, 1),
            self.box,
            ContactLocation(ContactType.VERTEX, 4),
            self.friction_coeff,
        )
        self.box_finger = ContactPair2d(
            "contact_2",
            self.box,
            ContactLocation(ContactType.FACE, 2),
            self.finger,
            ContactLocation(ContactType.VERTEX, 1),
            self.friction_coeff,
        )

        self.contact_pairs = [self.table_box, self.box_finger]

    def __post_init__(
        self,
    ) -> None:

        self.bodies = [self.table, self.box, self.finger]

        self._create_contact_pairs()

        self.contact_scene = ContactScene2d(self.bodies, self.contact_pairs, self.table)

        self.static_equilibrium_constraints = (
            self.contact_scene.create_static_equilibrium_constraints_for_body(self.box)
        )

        self.constraints = [
            cp.create_constraints(ContactMode.ROLLING) for cp in self.contact_pairs
        ]

        self.friction_cone_constraints = np.concatenate(
            [c.friction_cone for c in self.constraints]
        )

        self.relaxed_so_2_constraints = np.array(
            [c.relaxed_so_2 for c in self.constraints]
        )

        self.equal_contact_point_constraints = [
            c.equal_contact_points for c in self.constraints
        ]

        self.equal_rel_position_constraints = [
            c.equal_relative_positions for c in self.constraints
        ]

        breakpoint()
        self.equal_and_opposite_forces_constraints = [
            self.constraints[0].equal_and_opposite_forces
            # c.equal_and_opposite_forces for c in self.constraints # FIX: I must first add support for point contacts, otherwise it is impossible to satisfy friction cone constraints
        ]

        # FIX:
        # Plan:
        #  - Fix the following:

        # For each contact pair:
        #   add so2 constraints
        #   add nonpenetration constraints
        #
        #   add newtons 3rd law constraints
        #   add eq contact point constraints
        #
        # for each body:
        #   add friction cone constraints to all force
        #   add force minimization
        #
        #   if unactuated:
        #       add static eq constraints
        #

        # - Add support for point contacts

        self.p_TB_T = self.table_box.p_AB_A
        self.p_WB_W = self.p_TB_T

        self.p_BT_B = self.table_box.p_BA_B
        self.p_BW_B = self.p_BT_B

        # TODO: add in finger position relative to box

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

        # Non-penetration constraint
        # NOTE:! These are frame-dependent, keep in mind when generalizing these
        # Add nonpenetration constraint in table frame
        table_a_T = self.table.face_1.a
        pm1_B = self.box.v1
        pm3_B = self.box.v3

        self.nonpen_1_T = table_a_T.T.dot(self.R_TB).dot(pm1_B - self.pc1_B)[0, 0] >= 0
        self.nonpen_2_T = table_a_T.T.dot(self.R_TB).dot(pm3_B - self.pc1_B)[0, 0] >= 0

        # SO2 relaxation tightening
        # NOTE:! These are frame-dependent, keep in mind when generalizing these
        # Add SO(2) tightening
        cut_p1 = np.array([0, 1]).reshape((-1, 1))
        cut_p2 = np.array([1, 0]).reshape((-1, 1))
        a_cut, b_cut = construct_2d_plane_from_points(cut_p1, cut_p2)
        temp = self.R_TB[:, 0:1]
        self.so2_cut = (a_cut.T.dot(temp) - b_cut)[0, 0] >= 0

        # Quadratic cost on force
        self.forces_squared = (
            self.fc1_B.T.dot(self.fc1_B)[0, 0] + self.fc2_B.T.dot(self.fc2_B)[0, 0]
        )

    @property
    def variables(self) -> NpVariableArray:
        return self.contact_scene.variables


# TODO:: Generalize this. For now I moved all of this into a class for slightly easier data handling for plotting etc
@dataclass
class BoxFlipupDemo:
    use_quadratic_cost: bool = True
    use_moment_balance: bool = True
    use_friction_cone_constraint: bool = True
    use_force_balance_constraint: bool = True
    use_so2_constraint: bool = False
    use_so2_cut: bool = True
    use_non_penetration_constraint: bool = True
    use_equal_contact_point_constraint: bool = True
    use_equal_rel_position_constraint: bool = (
        False  # NOTE: this does not make the relaxation any tighter
    )
    use_newtons_third_law_constraint: bool = True

    def __init__(self) -> None:
        self._setup_bodies()
        self._setup_ctrl_points()
        self._setup_box_flipup_prog()

    def _setup_bodies(self) -> None:
        BOX_WIDTH = 0.3
        BOX_HEIGHT = 0.2

        TABLE_HEIGHT = 0.5
        TABLE_WIDTH = 2

        BOX_MASS = 1

        FRICTION_COEFF = 0.7

        self.finger = Point2d(actuated=True, name="finger", mass=None)
        self.friction_coeff = FRICTION_COEFF
        self.box = Box2d(
            actuated=False,
            name="box",
            mass=BOX_MASS,
            width=BOX_WIDTH,
            height=BOX_HEIGHT,
        )
        self.table = Box2d(
            actuated=True,
            name="table",
            mass=None,
            width=TABLE_WIDTH,
            height=TABLE_HEIGHT,
        )

        self.bodies = [self.box, self.table, self.finger]

    def _setup_ctrl_points(self) -> None:
        self.ctrl_points = [
            BoxFlipupCtrlPoint(
                self.table, self.box, self.finger, self.friction_coeff, idx
            )
            for idx in range(NUM_CTRL_POINTS)
        ]

    @property
    def R_WBs(self) -> List[npt.NDArray[Union[sym.Expression, sym.Variable]]]:  # type: ignore
        return [cp.R_WB for cp in self.ctrl_points]

    def _setup_box_flipup_prog(self) -> None:
        self.prog = MathematicalProgram()

        # TODO: this should be cleaned up
        MAX_FORCE = 10  # only used for mccorimick constraints
        variable_bounds = {
            "contact_1_box_c_n": (0.0, MAX_FORCE),
            "contact_1_box_c_f": (
                -self.friction_coeff * MAX_FORCE,
                self.friction_coeff * MAX_FORCE,
            ),
            "contact_1_table_c_n": (0.0, MAX_FORCE),
            "contact_1_table_c_f": (
                -self.friction_coeff * MAX_FORCE,
                self.friction_coeff * MAX_FORCE,
            ),
            "contact_1_table_lam": (0.0, 1.0),
            "contact_1_sin_th": (-1, 1),
            "contact_1_cos_th": (-1, 1),
            "contact_2_box_lam": (0.0, 1.0),
            "contact_2_box_c_n": (0, 3.8),
        }

        for ctrl_point in self.ctrl_points:
            self.prog.AddDecisionVariables(ctrl_point.variables)

            if self.use_friction_cone_constraint:
                self.prog.AddLinearConstraint(ctrl_point.friction_cone_constraints)

            if self.use_force_balance_constraint:
                self.prog.AddLinearConstraint(
                    ctrl_point.static_equilibrium_constraints.force_balance
                )

            if self.use_moment_balance:
                add_bilinear_constraints_to_prog(
                    ctrl_point.static_equilibrium_constraints.torque_balance,
                    self.prog,
                    variable_bounds,
                )

            if self.use_equal_contact_point_constraint:
                for c in ctrl_point.equal_contact_point_constraints:
                    add_bilinear_frame_constraints_to_prog(
                        c, self.prog, variable_bounds
                    )

            if self.use_equal_rel_position_constraint:
                for c in ctrl_point.equal_rel_position_constraints:
                    add_bilinear_frame_constraints_to_prog(
                        c, self.prog, variable_bounds
                    )

            if self.use_newtons_third_law_constraint:
                for c in ctrl_point.equal_and_opposite_forces_constraints:
                    add_bilinear_frame_constraints_to_prog(
                        c, self.prog, variable_bounds
                    )

            if self.use_so2_constraint:
                for c in ctrl_point.relaxed_so_2_constraints:
                    breakpoint()
                    self.prog.AddLorentzConeConstraint(1, c)  # type: ignore

            if self.use_non_penetration_constraint:
                self.prog.AddLinearConstraint(ctrl_point.nonpen_1_T)
                self.prog.AddLinearConstraint(ctrl_point.nonpen_2_T)

            if self.use_so2_cut:
                self.prog.AddLinearConstraint(ctrl_point.so2_cut)

            if self.use_quadratic_cost:
                self.prog.AddQuadraticCost(ctrl_point.forces_squared)

            else:  # Absolute value cost
                z = sym.Variable("z")
                self.prog.AddDecisionVariables(np.array([z]))
                self.prog.AddLinearCost(1 * z)

                for f in ctrl_point.fc1_B.flatten():
                    self.prog.AddLinearConstraint(z >= f)
                    self.prog.AddLinearConstraint(z >= -f)

                for f in ctrl_point.fc2_B.flatten():
                    self.prog.AddLinearConstraint(z >= f)
                    self.prog.AddLinearConstraint(z >= -f)

        # Initial and final condition
        th_initial = 0.0
        self._constrain_orientation_at_ctrl_point_idx(th_initial, 0)

        th_final = 0.9
        self._constrain_orientation_at_ctrl_point_idx(th_final, -1)

        # Don't allow contact position to change
        for idx in range(NUM_CTRL_POINTS - 1):
            self.prog.AddLinearConstraint(
                eq(self.ctrl_points[idx].pc2_B, self.ctrl_points[idx + 1].pc2_B)
            )
            self.prog.AddLinearConstraint(
                eq(self.ctrl_points[idx].pc1_T, self.ctrl_points[idx + 1].pc1_T)
            )

    def _constrain_orientation_at_ctrl_point_idx(self, theta: float, idx: int) -> None:
        condition = eq(self.R_WBs[idx], _angle_to_2d_rot_matrix(theta))
        for c in condition:
            self.prog.AddLinearConstraint(c)

    def solve(self) -> None:
        self.result = Solve(self.prog)
        print(f"Solution result: {self.result.get_solution_result()}")
        assert self.result.is_success()

        print(f"Cost: {self.result.get_optimal_cost()}")

    def get_force_balance_violation(self) -> npt.NDArray[np.float64]:
        # NOTE: This uses the force balance with normalized rotation matrix (because of SO(2) relaxation)
        fb_violation = np.hstack(
            [
                evaluate_np_formulas_array(
                    cp.static_equilibrium_constraints.normalized_force_balance,
                    self.result,
                )
                for cp in self.ctrl_points
            ]
        )  # (dims, num_ctrl_points)
        return fb_violation

    def get_torque_balance_violation(self) -> npt.NDArray[np.float64]:
        torque_balance_violation = np.hstack(
            [
                evaluate_np_formulas_array(
                    cp.static_equilibrium_constraints.torque_balance, self.result
                )
                for cp in self.ctrl_points
            ]
        )  # (1, num_ctrl_points)
        return torque_balance_violation

    def get_equal_contact_point_violation(
        self,
    ) -> List[EvaluatedContactFrameConstraints]:
        violations = [
            cp.equal_contact_point_constraints.evaluate(self.result)
            for cp in self.ctrl_points
        ]
        return violations

    def get_newtons_third_law_violation(self) -> List[EvaluatedContactFrameConstraints]:
        violations = [
            cp.equal_and_opposite_forces_constraints.evaluate(self.result)
            for cp in self.ctrl_points
        ]
        return violations

    def get_equal_rel_position_violation(
        self,
    ) -> List[EvaluatedContactFrameConstraints]:
        violations = [
            cp.equal_rel_position_constraints.evaluate(self.result)
            for cp in self.ctrl_points
        ]
        return violations

    @property
    def contact_forces_in_W(self) -> List[npt.NDArray[Union[sym.Expression, sym.Variable]]]:  # type: ignore
        fc1_B_ctrl_points = np.hstack(
            [cp.R_WB.dot(cp.fc1_B) for cp in self.ctrl_points]
        )
        fc1_T_ctrl_points = np.hstack(
            [cp.fc1_T for cp in self.ctrl_points]
        )  # T and W are the same frames

        fc2_B_ctrl_points = np.hstack(
            [cp.R_WB.dot(cp.fc2_B) for cp in self.ctrl_points]
        )
        # fc2_F_ctrl_points = np.hstack([cp.fc2_F for cp in self.ctrl_points]) # TODO: need rotation for this

        return [
            fc1_B_ctrl_points,
            fc1_T_ctrl_points,
            fc2_B_ctrl_points,
        ]

    @property
    def contact_positions_in_W(self) -> List[NpExpressionArray]:
        pc1_B_ctrl_points_in_W = np.hstack(
            [cp.p_WB_W + cp.R_WB.dot(cp.pc1_B) for cp in self.ctrl_points]
        )
        pc1_T_ctrl_points_in_W = np.hstack(
            [cp.pc1_T for cp in self.ctrl_points]
        )  # T and W are the same frames

        pc2_B_ctrl_points_in_W = np.hstack(
            [cp.p_WB_W + cp.R_WB.dot(cp.pc2_B) for cp in self.ctrl_points]
        )

        # pc2_F_ctrl_points_in_W = np.hstack([cp.pc2_F for cp in self.ctrl_points]) # TODO: need to add rotation for this

        return [pc1_B_ctrl_points_in_W, pc1_T_ctrl_points_in_W, pc2_B_ctrl_points_in_W]  # type: ignore

    @property
    def gravitational_force_in_W(self) -> npt.NDArray[np.float64]:
        return np.hstack([cp.box.gravity_force_in_W for cp in self.ctrl_points])

    @property
    def box_com_in_W(self) -> npt.NDArray[sym.Expression]:  # type: ignore
        return np.hstack([cp.p_WB_W for cp in self.ctrl_points])

    # FIX: Naming
    @property
    def box_com_B_in_W(self) -> npt.NDArray[sym.Expression]:  # type: ignore
        return np.hstack([-cp.R_WB.dot(cp.p_BW_B) for cp in self.ctrl_points])

    @property
    def box_orientation(self) -> List[npt.NDArray[Union[sym.Expression, sym.Variable]]]:  # type: ignore
        return [cp.R_WB for cp in self.ctrl_points]


def plan_box_flip_up_newtons_third_law():
    CONTACT_COLOR = "brown1"
    GRAVITY_COLOR = "blueviolet"
    BOX_COLOR = "aquamarine4"
    TABLE_COLOR = "bisque3"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--animate", help="Show animation", action="store_true", default=False
    )
    parser.add_argument(
        "--analysis", help="Show analysis", action="store_true", default=False
    )
    args = parser.parse_args()
    show_animation = args.animate
    show_analysis = args.analysis

    prog = BoxFlipupDemo()
    prog.solve()

    contact_positions_ctrl_points = [
        evaluate_np_expressions_array(pos, prog.result)
        for pos in prog.contact_positions_in_W
    ]
    contact_forces_ctrl_points = [
        evaluate_np_expressions_array(force, prog.result)
        for force in prog.contact_forces_in_W
    ]

    if show_animation:

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

        box_com_ctrl_points = prog.result.GetSolution(prog.box_com_in_W)
        viz_box_com = VisualizationPoint2d.from_ctrl_points(
            box_com_ctrl_points, GRAVITY_COLOR
        )
        viz_gravitional_force = VisualizationForce2d.from_ctrl_points(
            prog.result.GetSolution(prog.box_com_in_W),
            prog.result.GetSolution(prog.gravitational_force_in_W),
            GRAVITY_COLOR,
        )

        orientation_ctrl_points = [
            evaluate_np_expressions_array(R, prog.result) for R in prog.box_orientation
        ]
        viz_box = VisualizationPolygon2d.from_ctrl_points(
            box_com_ctrl_points,
            orientation_ctrl_points,
            prog.ctrl_points[0].box,
            BOX_COLOR,
        )

        table_pos_ctrl_points = np.zeros((2, NUM_CTRL_POINTS))
        table_orientation_ctrl_points = [np.eye(2)] * NUM_CTRL_POINTS

        viz_table = VisualizationPolygon2d.from_ctrl_points(
            table_pos_ctrl_points,
            table_orientation_ctrl_points,
            prog.ctrl_points[0].table,
            TABLE_COLOR,
        )

        # FIX: Naming
        # box_com_B_ctrl_points = prog.result.GetSolution(prog.box_com_B_in_W)
        # viz_box_com_B = VisualizationPoint2d.from_ctrl_points(
        #     box_com_B_ctrl_points, "carrot"
        # )

        viz = Visualizer2d()
        viz.visualize(
            viz_contact_positions + [viz_box_com],
            viz_contact_forces + [viz_gravitional_force],
            [viz_box, viz_table],
        )

    if show_analysis:
        equal_contact_point_violation = prog.get_equal_contact_point_violation()
        equal_rel_position_violation = prog.get_equal_rel_position_violation()
        newtons_third_law_violation = prog.get_newtons_third_law_violation()

        create_newtons_third_law_analysis(
            equal_contact_point_violation,
            equal_rel_position_violation,
            newtons_third_law_violation,
        )

        fb_violation_ctrl_points = prog.get_force_balance_violation()
        mb_violation_ctrl_points = prog.get_torque_balance_violation()

        create_static_equilibrium_analysis(
            fb_violation_ctrl_points, mb_violation_ctrl_points
        )

        # TODO: Generalize so that I don't need to pass in names!
        create_force_plot(
            contact_forces_ctrl_points,
            [
                "fc1_B",
                "fc1_T",
                "fc2_B",
            ],
        )

        show_plots()


if __name__ == "__main__":
    plan_box_flip_up_newtons_third_law()
