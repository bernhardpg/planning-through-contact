import time
import tkinter as tk
from dataclasses import dataclass
from tkinter import Canvas
from typing import List, NamedTuple, TypeVar, Union

import numpy as np
import numpy.typing as npt
import pydrake.symbolic as sym  # type: ignore
from pydrake.math import eq
from pydrake.solvers import MathematicalProgram, MathematicalProgramResult, Solve

from convex_relaxation.mccormick import relax_bilinear_expression
from geometry.bezier import BezierCtrlPoints, BezierCurve
from geometry.box import Box2d, construct_2d_plane_from_points
from geometry.contact_point import ContactPoint
from geometry.utilities import cross_2d
from visualize.colors import COLORS, RGB

# TODO move to its own plotting library
# Plotting
PLOT_CENTER = np.array([200, 300]).reshape((-1, 1))
PLOT_SCALE = 50
FORCE_SCALE = 0.2


@dataclass
class VisualizationPoint:
    position_curve: npt.NDArray[np.float64]  # (N, dims)
    color: RGB

    @classmethod
    def from_ctrl_points(
        cls, ctrl_points: npt.NDArray[np.float64], color: str = "red1"
    ) -> "VisualizationPoint":
        position_curve = BezierCurve.create_from_ctrl_points(
            ctrl_points
        ).eval_entire_interval()
        return cls(position_curve, COLORS[color].hex_format)


@dataclass
class VisualizationForce(VisualizationPoint):
    force_curve: npt.NDArray[np.float64]  # (N, dims)

    @classmethod
    def from_ctrl_points(
        cls,
        ctrl_points_position: npt.NDArray[
            np.float64
        ],  # TODO create a struct or class for ctrl points
        ctrl_points_force: npt.NDArray[np.float64],
        color: str = "red1",
    ) -> "VisualizationPoint":
        position_curve = BezierCurve.create_from_ctrl_points(
            ctrl_points_position
        ).eval_entire_interval()
        force_curve = BezierCurve.create_from_ctrl_points(
            ctrl_points_force
        ).eval_entire_interval()
        return cls(position_curve, COLORS[color].hex_format, force_curve)


def _flatten_points(points):
    return list(points.flatten(order="F"))


def _make_plotable(
    points: npt.NDArray[np.float64],
    scale: float = PLOT_SCALE,
    center: float = PLOT_CENTER,
) -> List[float]:
    points_flipped_y_axis = np.vstack([points[0, :], -points[1, :]])
    points_transformed = points_flipped_y_axis * scale + center
    plotable_points = _flatten_points(points_transformed)
    return plotable_points


def _draw_force(
    pos_B: npt.NDArray[np.float64],
    force_B: npt.NDArray[np.float64],
    R_WB: npt.NDArray[np.float64],
    p_WB: npt.NDArray[np.float64],
    canvas,
    color: str = "#0f0",
):
    force_W = np.hstack([p_WB + R_WB.dot(pos_B), p_WB + R_WB.dot(pos_B + force_B)])
    force_points = _make_plotable(force_W)
    canvas.create_line(force_points, width=2, arrow=tk.LAST, fill=color)


def _draw_circle(
    pos_B: npt.NDArray[np.float64],
    R_WB: npt.NDArray[np.float64],
    p_WB: npt.NDArray[np.float64],
    canvas,
    radius: float = 0.1,
    color: str = "#e28743",
) -> None:
    pos_W = p_WB + R_WB.dot(pos_B)
    lower_left = pos_W - radius
    upper_right = pos_W + radius
    points = _make_plotable(np.hstack([lower_left, upper_right]))
    canvas.create_oval(points[0], points[1], points[2], points[3], fill=color, width=0)


############


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


def _create_rot_matrix_from_ctrl_point_idx(
    cos_ths: npt.NDArray[T],  # type: ignore
    sin_ths: npt.NDArray[T],  # type: ignore
    ctrl_point_idx: int,
) -> npt.NDArray[T]:  # type: ignore
    cos_th, sin_th = cos_ths[0, ctrl_point_idx], sin_ths[0, ctrl_point_idx]
    rot_matrix = np.array([[cos_th, -sin_th], [sin_th, cos_th]])
    return rot_matrix


# NOTE that this does not use the ContactPoint defined in the library
# TODO this should be unified


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

        # TODO use enums instead
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


# TODO should be in a file with boxflipup demo
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

        box = Box2d(BOX_WIDTH, BOX_HEIGHT)
        table = Box2d(TABLE_WIDTH, TABLE_HEIGHT)
        finger = Box2d(FINGER_WIDTH, FINGER_HEIGHT)

        self.table_box = ContactPair(
            "contact_1",
            table,
            "face_1",
            "table",
            box,
            "corner_4",
            "box",
            FRICTION_COEFF,
        )
        self.box_finger = ContactPair(
            "contact_2",
            box,
            "face_2",
            "box",
            finger,
            "corner_1",
            "finger",
            FRICTION_COEFF,
        )

        # Constant variables
        self.fg_W = np.array([0, -BOX_MASS * GRAV_ACC]).reshape((-1, 1))

        self.p_TB = self.table_box.p_AB_A
        self.R_TB = self.table_box.R_AB

        self.cos_th = self.R_TB[0, 0]
        self.sin_th = self.R_TB[1, 0]
        self.R_WB = self.R_TB  # World frame is the same as table frame

        self.fc1_B = self.table_box.contact_point_B.contact_force
        self.fc1_T = self.table_box.contact_point_A.contact_force
        self.pc1_B = self.table_box.contact_point_B.contact_position
        self.pc1_T = self.table_box.contact_point_A.contact_position

        self.fc2_B = self.box_finger.contact_point_A.contact_force
        self.fc2_F = self.box_finger.contact_point_B.contact_force
        self.pc2_B = self.box_finger.contact_point_A.contact_position
        self.pc2_F = self.box_finger.contact_point_B.contact_position

        self.sum_of_forces_B = self.fc1_B + self.fc2_B + self.R_WB.T.dot(self.fg_W)  # type: ignore

        # TODO clean up, this is just for plotting rounding error!
        self.R_WB_normalized = self.R_WB / sym.sqrt(
            (self.cos_th**2 + self.sin_th**2)
        )
        self.true_sum_of_forces_B = self.fc1_B + self.fc2_B + self.R_WB_normalized.T.dot(self.fg_W)  # type: ignore

        # Add moment balance using a linear relaxation
        # TODO now this is hardcoded, in the future this should be automatically added for all unactuated objects
        self.sum_of_moments_B = cross_2d(self.pc1_B, self.fc1_B) + cross_2d(
            self.pc2_B, self.fc2_B
        )

        self.relaxed_so_2_constraint = (self.R_WB.T.dot(self.R_WB))[0, 0]

        # Non-penetration constraint
        # NOTE! These are frame-dependent, keep in mind when generalizing these
        # Add nonpenetration constraint in table frame
        table_a_T = table.a1[0]
        pm1_B = box.p1
        pm3_B = box.p3

        self.nonpen_1_T = table_a_T.T.dot(self.R_TB).dot(pm1_B - self.pc1_B)[0, 0] >= 0
        self.nonpen_2_T = table_a_T.T.dot(self.R_TB).dot(pm3_B - self.pc1_B)[0, 0] >= 0

        # SO2 relaxation tightening
        # NOTE! These are frame-dependent, keep in mind when generalizing these
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


# TODO: Generalize this. For now I moved all of this into a class for slightly easier data handling for plotting etc
@dataclass
class BoxFlipupDemo:
    friction_coeff: float = 0.7 # TODO unify this with ctrl point constants
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

        # TODO this should be cleaned up
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
                # TODO this is duplicated code, clean up!
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
    def forces(self) -> npt.NDArray[Union[sym.Expression, sym.Variable]]:  # type: ignore
        return np.hstack(self.ctrl_points[0].fc1_B)


def plan_box_flip_up_newtons_third_law():
    prog = BoxFlipupDemo()
    breakpoint
    prog.solve()

    fb_violation = prog.get_force_balance_violation()
    mb_violation = prog.get_moment_balance_violation()

    breakpoint()

    force_test = BezierCtrlPoints(prog.result.GetSolution(prog.forces))

    breakpoint()

    cos_th_vals = result.GetSolution(cos_ths).reshape((1, -1))
    sin_th_vals = result.GetSolution(sin_ths).reshape((1, -1))
    R_WB_vals = [
        _create_rot_matrix_from_ctrl_point_idx(cos_th_vals, sin_th_vals, idx)
        for idx in range(cos_th_vals.shape[1])
    ]

    #######
    # Animation

    sin_curve = BezierCurve.create_from_ctrl_points(sin_th_vals).eval_entire_interval()
    cos_curve = BezierCurve.create_from_ctrl_points(cos_th_vals).eval_entire_interval()
    R_curve = [
        np.array([[c[0], -s[0]], [s[0], c[0]]]) for c, s in zip(cos_curve, sin_curve)
    ]

    # TODO helper function that should ideally be used other places too!
    def _get_trajectory_from_solution(
        expressions: List[Union[sym.Variable, sym.Expression]],
        result: MathematicalProgramResult,
    ) -> npt.NDArray[np.float64]:

        from_expr_to_float = np.vectorize(lambda expr: expr.Evaluate())
        ctrl_points = np.hstack([from_expr_to_float(result.GetSolution(e)) for e in expressions])  # type: ignore
        curve = BezierCurve.create_from_ctrl_points(ctrl_points).eval_entire_interval()
        return curve

    fc1_B_curve = _get_trajectory_from_solution(fc1_Bs, result)
    fc1_T_curve = _get_trajectory_from_solution(fc1_Ts, result)
    pc1_B_curve = _get_trajectory_from_solution(pc1_Bs, result)
    pc1_T_curve = _get_trajectory_from_solution(pc1_Ts, result)
    fc2_B_curve = _get_trajectory_from_solution(fc2_Bs, result)
    fc2_F_curve = _get_trajectory_from_solution(fc2_Fs, result)
    pc2_B_curve = _get_trajectory_from_solution(pc2_Bs, result)
    pc2_F_curve = _get_trajectory_from_solution(pc2_Fs, result)

    p_TB_curve = BezierCurve.create_from_ctrl_points(
        np.hstack([result.GetSolution(step).reshape((-1, 1)) for step in p_TBs])
    ).eval_entire_interval()

    pm4_W = np.array([0, TABLE_HEIGHT / 2]).reshape((-1, 1))  # TODO investigate this
    pc1_B = pc1_B_curve[
        0:1, :
    ].T  # NOTE: This stays fixed for the entire interval, due to being a corner! This is a quick fix
    p_WB_curve = [pm4_W - R.dot(pc1_B) for R in R_curve]

    # Helper transforms
    R_WT = np.eye(2)  # there is no rotation from table to world frame
    p_WT = np.array([0, 0]).reshape((-1, 1))
    R_WF = np.eye(2)  # same for finger
    p_WF = np.array([0, 0]).reshape(
        (-1, 1)
    )  # TODO This is wrong. This should be a decision variable!

    ## Plotting

    app = tk.Tk()
    app.title("Box")

    canvas = Canvas(app, width=500, height=500, bg="white")
    canvas.pack()

    # TODO clean up this code!
    f_Wg_val = fg_W * FORCE_SCALE
    for idx in range(len(cos_curve)):

        canvas.delete("all")
        R_WB_val = R_curve[idx]
        det_R = np.linalg.det(R_WB_val)

        p_WB = p_WB_curve[idx]

        fc1_B_val = fc1_B_curve[idx].reshape((-1, 1)) * FORCE_SCALE
        pc1_B_val = pc1_B_curve[idx].reshape((-1, 1))

        fc2_B_val = fc2_B_curve[idx].reshape((-1, 1)) * FORCE_SCALE
        pc2_B_val = pc2_B_curve[idx].reshape((-1, 1))

        fc1_T_val = fc1_T_curve[idx].reshape((-1, 1)) * FORCE_SCALE
        pc1_T_val = pc1_T_curve[idx].reshape((-1, 1))

        fc2_F_val = fc2_F_curve[idx].reshape((-1, 1)) * FORCE_SCALE
        pc2_F_val = pc2_F_curve[idx].reshape((-1, 1))

        points_box = _make_plotable(p_WB + R_WB_val.dot(box.corners))
        points_table = _make_plotable(table.corners)

        canvas.create_polygon(points_box, fill="#88f")
        canvas.create_polygon(points_table, fill="#2f2f2f")

        _draw_force(pc1_B_val, fc1_B_val, R_WB_val, p_WB, canvas, "#0f0")
        _draw_force(pc2_B_val, fc2_B_val, R_WB_val, p_WB, canvas, "#0f0")

        _draw_force(pc2_F_val, fc2_F_val, R_WB_val.T, p_WB, canvas, "#ff3")

        _draw_force(pc1_T_val, fc1_T_val, R_WT, p_WT, canvas, "#ff3")

        grav_force = _make_plotable(np.hstack([p_WB, p_WB + f_Wg_val * det_R]))
        canvas.create_line(grav_force, width=2, arrow=tk.LAST, fill="#f00")

        # Plot contact locations
        _draw_circle(pc1_B_val, R_WB_val, p_WB, canvas)
        _draw_circle(pc1_T_val, R_WT, p_WT, canvas, color="#1e81b0")

        _draw_circle(pc2_B_val, R_WB_val, p_WB, canvas)

        # _draw_circle(pc2_F_val, R_WF, p_WF, canvas, color="#1e81b0") # finger location is constant in finger frame

        p_TB_val = p_TB_curve[idx].reshape((-1, 1))
        _draw_circle(p_TB_val, np.eye(2), np.array([[0], [0]]), canvas, color="#1e8dff")

        canvas.update()
        time.sleep(0.05)
        if idx == 0:
            time.sleep(2)

    app.mainloop()


if __name__ == "__main__":
    plan_box_flip_up_newtons_third_law()
