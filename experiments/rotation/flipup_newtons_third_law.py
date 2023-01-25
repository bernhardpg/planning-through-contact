from dataclasses import dataclass
from typing import List, TypeVar, Union

import numpy as np
import numpy.typing as npt
import pydrake.symbolic as sym  # type: ignore
from pydrake.math import eq
from pydrake.solvers import MathematicalProgram, Solve

from convex_relaxation.mccormick import relax_bilinear_expression
from geometry.box import Box2d, construct_2d_plane_from_points
from geometry.utilities import cross_2d
from visualize.visualizer_2d import (
    VisualizationForce2d,
    VisualizationPoint2d,
    VisualizationPolygon2d,
    Visualizer2d,
)


def _convert_formula_to_lhs_expression(form: sym.Formula) -> sym.Expression:
    lhs, rhs = form.Unapply()[1]  # type: ignore
    expr = lhs - rhs
    return expr


T = TypeVar("T")


def _angle_to_2d_rot_matrix(theta: float) -> npt.NDArray[np.float64]:
    rot_matrix = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    return rot_matrix


# NOTE: that this does not use the ContactPoint defined in the library
# TODO: this should be unified
class ContactPoint2d:
    def __init__(
        self,
        body: Box2d,
        contact_location: str,
        friction_coeff: float = 0.5,
        name: str = "unnamed",
    ) -> None:
        self.normal_vec, self.tangent_vec = body.get_norm_and_tang_vecs_from_location(
            contact_location
        )
        self.friction_coeff = friction_coeff

        self.normal_force = sym.Variable(f"{name}_c_n")
        self.friction_force = sym.Variable(f"{name}_c_f")

        # TODO: use enums instead
        if "face" in contact_location:
            self.contact_type = "face"
        elif "corner" in contact_location:
            self.contact_type = "corner"
        else:
            raise ValueError(f"Unsupported contact location: {contact_location}")

        if self.contact_type == "face":
            self.lam = sym.Variable(f"{name}_lam")
            vertices = body.get_proximate_vertices_from_location(contact_location)
            self.contact_position = (
                self.lam * vertices[0] + (1 - self.lam) * vertices[1]
            )
        else:
            corner_vertex = body.get_proximate_vertices_from_location(contact_location)
            self.contact_position = corner_vertex

    @property
    def contact_force(self) -> npt.NDArray[sym.Expression]:  # type: ignore
        return (
            self.normal_force * self.normal_vec + self.friction_force * self.tangent_vec
        )

    @property
    def variables(self) -> npt.NDArray[sym.Variable]:  # type: ignore
        if self.contact_type == "face":
            return np.array([self.normal_force, self.friction_force, self.lam])
        else:
            return np.array([self.normal_force, self.friction_force])

    def create_friction_cone_constraints(self) -> npt.NDArray[sym.Formula]:  # type: ignore
        upper_bound = self.friction_force <= self.friction_coeff * self.normal_force
        lower_bound = -self.friction_coeff * self.normal_force <= self.friction_force
        normal_force_positive = self.normal_force >= 0
        return np.vstack([upper_bound, lower_bound, normal_force_positive])


@dataclass
class ContactPair:
    def __init__(
        self,
        pair_name: str,
        body_A: Box2d,
        contact_location_A: str,
        name_A: str,
        body_B: Box2d,
        contact_location_B: str,
        name_B: str,
        friction_coeff: float,
    ) -> None:
        self.contact_point_A = ContactPoint2d(
            body_A,
            contact_location_A,
            friction_coeff,
            name=f"{pair_name}_{name_A}",
        )
        self.contact_point_B = ContactPoint2d(
            body_B,
            contact_location_B,
            friction_coeff,
            name=f"{pair_name}_{name_B}",
        )

        cos_th = sym.Variable(f"{pair_name}_cos_th")
        sin_th = sym.Variable(f"{pair_name}_sin_th")
        self.R_AB = np.array([[cos_th, -sin_th], [sin_th, cos_th]])

        p_AB_A_x = sym.Variable(f"{pair_name}_p_AB_A_x")
        p_AB_A_y = sym.Variable(f"{pair_name}_p_AB_A_y")
        self.p_AB_A = np.array([p_AB_A_x, p_AB_A_y]).reshape((-1, 1))

        p_BA_B_x = sym.Variable(f"{pair_name}_p_BA_B_x")
        p_BA_B_y = sym.Variable(f"{pair_name}_p_BA_B_y")
        self.p_BA_B = np.array([p_BA_B_x, p_BA_B_y]).reshape((-1, 1))

    @property
    def variables(self) -> npt.NDArray[sym.Variable]:  # type: ignore
        return np.concatenate(
            [
                self.contact_point_A.variables,
                self.contact_point_B.variables,
                self.p_AB_A.flatten(),
                self.p_BA_B.flatten(),
            ]
        )

    def create_equal_contact_point_constraints(self) -> npt.NDArray[sym.Formula]:  # type: ignore
        p_Ac_A = self.contact_point_A.contact_position
        p_Bc_B = self.contact_point_B.contact_position

        p_Bc_A = self.R_AB.dot(p_Bc_B)
        constraints_in_frame_A = eq(p_Ac_A, self.p_AB_A + p_Bc_A)

        p_Ac_B = self.R_AB.T.dot(p_Ac_A)
        constraints_in_frame_B = eq(p_Bc_B, self.p_BA_B + p_Ac_B)

        rel_pos_equal_in_A = eq(self.p_AB_A, -self.R_AB.dot(self.p_BA_B))
        rel_pos_equal_in_B = eq(self.p_BA_B, -self.R_AB.T.dot(self.p_AB_A))

        return np.vstack(
            (
                constraints_in_frame_A,
                constraints_in_frame_B,
                rel_pos_equal_in_A,
                rel_pos_equal_in_B,
            )
        )

    def create_newtons_third_law_force_constraints(self) -> npt.NDArray[sym.Formula]:  # type: ignore
        f_c_A = self.contact_point_A.contact_force
        f_c_B = self.contact_point_B.contact_force

        equal_and_opposite_in_A = eq(f_c_A, -self.R_AB.dot(f_c_B))
        equal_and_opposite_in_B = eq(f_c_B, -self.R_AB.T.dot(f_c_A))

        return np.vstack(
            (
                equal_and_opposite_in_A,
                equal_and_opposite_in_B,
            )
        )


# TODO: should be in a file with boxflipup demo
class BoxFlipupCtrlPoint:
    table_box: ContactPair
    box_finger: ContactPair

    def __init__(self) -> None:
        BOX_WIDTH = 3
        BOX_HEIGHT = 2

        TABLE_HEIGHT = 2
        TABLE_WIDTH = 10

        FINGER_HEIGHT = 1
        FINGER_WIDTH = 1

        BOX_MASS = 1
        GRAV_ACC = 9.81

        FRICTION_COEFF = 0.7

        self.box = Box2d(BOX_WIDTH, BOX_HEIGHT)
        self.table = Box2d(TABLE_WIDTH, TABLE_HEIGHT)
        self.finger = Box2d(FINGER_WIDTH, FINGER_HEIGHT)

        self.table_box = ContactPair(
            "contact_1",
            self.table,
            "face_1",
            "table",
            self.box,
            "corner_4",
            "box",
            FRICTION_COEFF,
        )
        self.box_finger = ContactPair(
            "contact_2",
            self.box,
            "face_2",
            "box",
            self.finger,
            "corner_1",
            "finger",
            FRICTION_COEFF,
        )

        # Constant variables
        self.fg_W = np.array([0, -BOX_MASS * GRAV_ACC]).reshape((-1, 1))

        self.p_TB_T = self.table_box.p_AB_A
        self.p_WB_W = self.p_TB_T

        self.p_BT_B = self.table_box.p_BA_B
        self.p_BW_B = self.p_BT_B

        # TODO: add in finger position relative to box

        self.R_TB = self.table_box.R_AB
        self.R_WB = self.R_TB  # World frame is the same as table frame

        self.cos_th = self.R_TB[0, 0]
        self.sin_th = self.R_TB[1, 0]

        self.fc1_B = self.table_box.contact_point_B.contact_force
        self.fc1_T = self.table_box.contact_point_A.contact_force
        self.pc1_B = self.table_box.contact_point_B.contact_position
        self.pc1_T = self.table_box.contact_point_A.contact_position

        self.fc2_B = self.box_finger.contact_point_A.contact_force
        self.fc2_F = self.box_finger.contact_point_B.contact_force
        self.pc2_B = self.box_finger.contact_point_A.contact_position
        self.pc2_F = self.box_finger.contact_point_B.contact_position

        self.sum_of_forces_B = self.fc1_B + self.fc2_B + self.R_WB.T.dot(self.fg_W)  # type: ignore

        # TODO: clean up, this is just for plotting rounding error!
        self.R_WB_normalized = self.R_WB / sym.sqrt(
            (self.cos_th**2 + self.sin_th**2)
        )
        self.true_sum_of_forces_B = self.fc1_B + self.fc2_B + self.R_WB_normalized.T.dot(self.fg_W)  # type: ignore

        # Add moment balance using a linear relaxation
        # TODO: now this is hardcoded, in the future this should be automatically added for all unactuated objects
        self.sum_of_moments_B = cross_2d(self.pc1_B, self.fc1_B) + cross_2d(
            self.pc2_B, self.fc2_B
        )

        self.relaxed_so_2_constraint = (self.R_WB.T.dot(self.R_WB))[0, 0]

        # Non-penetration constraint
        # NOTE:! These are frame-dependent, keep in mind when generalizing these
        # Add nonpenetration constraint in table frame
        table_a_T = self.table.a1[0]
        pm1_B = self.box.p1
        pm3_B = self.box.p3

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


# TODO:: Generalize this. For now I moved all of this into a class for slightly easier data handling for plotting etc
@dataclass
class BoxFlipupDemo:
    friction_coeff: float = 0.7  # TODO: unify this with ctrl point constants
    use_quadratic_cost: bool = True
    use_moment_balance: bool = True
    use_friction_cone_constraint: bool = True
    use_force_balance_constraint: bool = True
    use_so2_constraint: bool = True
    use_so2_cut: bool = True
    use_non_penetration_constraint: bool = True
    use_equal_contact_point_constraint: bool = True
    use_newtons_third_law_constraint: bool = True

    def __init__(self) -> None:
        self._setup_ctrl_points()
        self._setup_box_flipup_prog()

    def _setup_ctrl_points(self) -> None:
        NUM_CTRL_POINTS = 3
        self.ctrl_points = [BoxFlipupCtrlPoint() for _ in range(NUM_CTRL_POINTS)]

    @property
    def R_WBs(self) -> List[npt.NDArray[Union[sym.Expression, sym.Variable]]]:  # type: ignore
        return [cp.R_WB for cp in self.ctrl_points]

    @property
    def force_balances(self) -> List[npt.NDArray[sym.Expression]]:  # type: ignore
        return [cp.true_sum_of_forces_B for cp in self.ctrl_points]

    @property
    def moment_balances(self) -> List[sym.Expression]:  # type: ignore
        return [cp.sum_of_moments_B for cp in self.ctrl_points]

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
            "contact_2_box_c_n": (0, MAX_FORCE),
        }

        for ctrl_point in self.ctrl_points:
            self.prog.AddDecisionVariables(
                np.array([ctrl_point.cos_th, ctrl_point.sin_th])
            )
            self.prog.AddDecisionVariables(ctrl_point.table_box.variables)
            self.prog.AddDecisionVariables(ctrl_point.box_finger.variables)

            if self.use_friction_cone_constraint:
                self.prog.AddLinearConstraint(
                    ctrl_point.table_box.contact_point_A.create_friction_cone_constraints()
                )
                self.prog.AddLinearConstraint(
                    ctrl_point.table_box.contact_point_B.create_friction_cone_constraints()
                )
                self.prog.AddLinearConstraint(
                    ctrl_point.box_finger.contact_point_A.create_friction_cone_constraints()
                )

            if self.use_force_balance_constraint:
                self.prog.AddLinearConstraint(eq(ctrl_point.sum_of_forces_B, 0))

            if self.use_moment_balance:
                # This is hardcoded for now and should be generalized. It is used to define the mccormick envelopes
                (
                    relaxed_sum_of_moments,
                    new_vars,
                    mccormick_envelope_constraints,
                ) = relax_bilinear_expression(
                    ctrl_point.sum_of_moments_B, variable_bounds
                )
                self.prog.AddDecisionVariables(new_vars)  # type: ignore
                self.prog.AddLinearConstraint(relaxed_sum_of_moments == 0)
                for c in mccormick_envelope_constraints:
                    self.prog.AddLinearConstraint(c)

            if self.use_equal_contact_point_constraint:
                equal_contact_point_constraints = (
                    ctrl_point.table_box.create_equal_contact_point_constraints()
                )

                for row in equal_contact_point_constraints:
                    for constraint in row:
                        expr = _convert_formula_to_lhs_expression(constraint)
                        expression_is_linear = sym.Polynomial(expr).TotalDegree() == 1
                        if expression_is_linear:
                            self.prog.AddLinearConstraint(expr == 0)
                        else:
                            (
                                relaxed_expr,
                                new_vars,
                                mccormick_envelope_constraints,
                            ) = relax_bilinear_expression(expr, variable_bounds)
                            self.prog.AddDecisionVariables(new_vars)  # type: ignore
                            self.prog.AddLinearConstraint(relaxed_expr == 0)
                            for c in mccormick_envelope_constraints:
                                self.prog.AddLinearConstraint(c)

            if self.use_newtons_third_law_constraint:
                newtons_third_law_constraints = (
                    ctrl_point.table_box.create_newtons_third_law_force_constraints()
                )
                # TODO: this is duplicated code, clean up!
                for row in newtons_third_law_constraints:
                    for constraint in row:
                        expr = _convert_formula_to_lhs_expression(constraint)
                        expression_is_linear = sym.Polynomial(expr).TotalDegree() == 1
                        if expression_is_linear:
                            self.prog.AddLinearConstraint(expr == 0)
                        else:
                            (
                                relaxed_expr,
                                new_vars,
                                mccormick_envelope_constraints,
                            ) = relax_bilinear_expression(expr, variable_bounds)
                            self.prog.AddDecisionVariables(new_vars)  # type: ignore
                            self.prog.AddLinearConstraint(relaxed_expr == 0)
                            for c in mccormick_envelope_constraints:
                                self.prog.AddLinearConstraint(c)

            if self.use_so2_constraint:
                self.prog.AddLorentzConeConstraint(1, ctrl_point.relaxed_so_2_constraint)  # type: ignore

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
        self.prog.AddLinearConstraint(
            eq(self.ctrl_points[0].pc2_B, self.ctrl_points[1].pc2_B)
        )
        self.prog.AddLinearConstraint(
            eq(self.ctrl_points[1].pc2_B, self.ctrl_points[2].pc2_B)
        )

        self.prog.AddLinearConstraint(
            eq(self.ctrl_points[0].pc1_T, self.ctrl_points[1].pc1_T)
        )
        self.prog.AddLinearConstraint(
            eq(self.ctrl_points[1].pc1_T, self.ctrl_points[2].pc1_T)
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
        fb_violation = self._get_solution_for_np_exprs(self.force_balances)
        return fb_violation

    def get_moment_balance_violation(self) -> npt.NDArray[np.float64]:
        return self._get_solution_for_exprs(self.moment_balances)

    def _get_solution_for_np_expr(self, expr: npt.NDArray[sym.Expression]) -> npt.NDArray[np.float64]:  # type: ignore
        from_expr_to_float = np.vectorize(lambda expr: expr.Evaluate())
        solutions = from_expr_to_float(self.result.GetSolution(expr))
        return solutions

    def _get_solution_for_np_exprs(self, exprs: List[npt.NDArray[sym.Expression]]) -> npt.NDArray[np.float64]:  # type: ignore
        from_expr_to_float = np.vectorize(lambda expr: expr.Evaluate())
        solutions = np.array(
            [from_expr_to_float(self.result.GetSolution(e).flatten()) for e in exprs]
        )
        return solutions

    def _get_solution_for_exprs(self, expr: List[sym.Expression]) -> npt.NDArray[np.float64]:  # type: ignore
        solutions = np.array([self.result.GetSolution(e).Evaluate() for e in expr])
        return solutions

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

        return [fc1_B_ctrl_points, fc1_T_ctrl_points, fc2_B_ctrl_points]

    @property
    def contact_positions_in_W(self) -> List[npt.NDArray[sym.Expression]]:  # type: ignore
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
    def gravitational_force_in_W(self) -> npt.NDArray[sym.Expression]:  # type: ignore
        return np.hstack([cp.fg_W for cp in self.ctrl_points])

    @property
    def box_com_in_W(self) -> npt.NDArray[sym.Expression]:  # type: ignore
        return np.hstack([cp.p_WB_W for cp in self.ctrl_points])

    @property
    def box_orientation(self) -> List[npt.NDArray[Union[sym.Expression, sym.Variable]]]:  # type: ignore
        return [cp.R_WB for cp in self.ctrl_points]


def plan_box_flip_up_newtons_third_law():
    CONTACT_COLOR = "brown1"
    GRAVITY_COLOR = "blueviolet"
    BOX_COLOR = "aquamarine4"
    TABLE_COLOR = "bisque3"

    NUM_CTRL_POINTS = 3

    prog = BoxFlipupDemo()
    prog.solve()

    contact_positions_ctrl_points = [
        prog._get_solution_for_np_expr(pos) for pos in prog.contact_positions_in_W
    ]
    contact_forces_ctrl_points = [
        prog._get_solution_for_np_expr(force) for force in prog.contact_forces_in_W
    ]

    viz_contact_positions = [
        VisualizationPoint2d.from_ctrl_points(pos, CONTACT_COLOR)
        for pos in contact_positions_ctrl_points
    ]
    viz_contact_forces = [
        VisualizationForce2d.from_ctrl_points(pos, force, CONTACT_COLOR)
        for pos, force in zip(contact_positions_ctrl_points, contact_forces_ctrl_points)
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
        prog._get_solution_for_np_expr(R) for R in prog.box_orientation
    ]
    viz_box = VisualizationPolygon2d.from_ctrl_points(
        box_com_ctrl_points, orientation_ctrl_points, prog.ctrl_points[0].box, BOX_COLOR
    )

    table_pos_ctrl_points = np.zeros((2, NUM_CTRL_POINTS))
    table_orientation_ctrl_points = [np.eye(2)] * NUM_CTRL_POINTS

    viz_table = VisualizationPolygon2d.from_ctrl_points(
        table_pos_ctrl_points,
        table_orientation_ctrl_points,
        prog.ctrl_points[0].table,
        TABLE_COLOR,
    )

    viz = Visualizer2d()
    viz.visualize(
        viz_contact_positions + [viz_box_com],
        viz_contact_forces + [viz_gravitional_force],
        [viz_box, viz_table],
    )

    # fb_violation = prog.get_force_balance_violation()
    # mb_violation = prog.get_moment_balance_violation()


if __name__ == "__main__":
    plan_box_flip_up_newtons_third_law()
