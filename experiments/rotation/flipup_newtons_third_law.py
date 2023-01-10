import time
from dataclasses import dataclass
from typing import List, TypeVar, Union

import numpy as np
import numpy.typing as npt
import pydrake.symbolic as sym  # type: ignore
from pydrake.math import eq
from pydrake.solvers import MathematicalProgram, MathematicalProgramResult, Solve

from convex_relaxation.mccormick import relax_bilinear_expression
from geometry.bezier import BezierCurve
from geometry.box import Box2d, construct_2d_plane_from_points
from geometry.contact_point import ContactPoint
from geometry.utilities import cross_2d

T = TypeVar("T")


def _create_rot_matrix(
    cos_ths: npt.NDArray[T],  # type: ignore
    sin_ths: npt.NDArray[T],  # type: ignore
    ctrl_point_idx: int,
) -> npt.NDArray[T]:  # type: ignore
    cos_th, sin_th = cos_ths[0, ctrl_point_idx], sin_ths[0, ctrl_point_idx]
    rot_matrix = np.array([[cos_th, -sin_th], [sin_th, cos_th]])
    return rot_matrix


# NOTE that this does not use the ContactPoint2D defined in the library
# TODO this should be unified


class ContactPoint2D:
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
        self.contact_point_A = ContactPoint2D(
            body_A,
            contact_location_A,
            friction_coeff,
            name=f"{pair_name}_{name_A}",
        )
        self.contact_point_B = ContactPoint2D(
            body_B,
            contact_location_B,
            friction_coeff,
            name=f"{pair_name}_{name_B}",
        )

    @property
    def variables(self) -> npt.NDArray[sym.Variable]:  # type: ignore
        return np.concatenate(
            [self.contact_point_A.variables, self.contact_point_B.variables]
        )


def plan_box_flip_up_newtons_third_law():
    USE_QUADRATIC_COST = True
    USE_MOMENT_BALANCE = True
    USE_FRICTION_CONE_CONSTRAINT = True
    USE_FORCE_BALANCE_CONSTRAINT = True
    USE_SO2_CONSTRAINT = True
    USE_SO2_CUT = True
    USE_NON_PENETRATION_CONSTRAINT = True

    NUM_CTRL_POINTS = 3

    BOX_WIDTH = 3
    BOX_HEIGHT = 2

    TABLE_HEIGHT = 2
    TABLE_WIDTH = 10

    FINGER_HEIGHT = 1
    FINGER_WIDTH = 1

    BOX_MASS = 1
    GRAV_ACC = 9.81
    FRICTION_COEFF = 0.7

    prog = MathematicalProgram()

    # FIRST STEP: Only one ctrl point
    # TODO: I can just run all of this code for each ctrl point
    cos_ths = []
    sin_ths = []

    fc1_Bs = []
    fc1_Ts = []
    pc1_Bs = []
    pc1_Ts = []

    fc2_Bs = []
    fc2_Fs = []
    pc2_Bs = []
    pc2_Fs = []

    for ctrl_point_idx in range(NUM_CTRL_POINTS):
        box = Box2d(BOX_WIDTH, BOX_HEIGHT)
        table = Box2d(TABLE_WIDTH, TABLE_HEIGHT)
        finger = Box2d(FINGER_WIDTH, FINGER_HEIGHT)

        cp_box_table = ContactPair(
            "contact_1",
            box,
            "corner_4",
            "box",
            table,
            "face_1",
            "table",
            FRICTION_COEFF,
        )
        cp_box_finger = ContactPair(
            "contact_2",
            box,
            "face_2",
            "box",
            finger,
            "corner_1",
            "finger",
            FRICTION_COEFF,
        )

        # Four reference frames: B (Box), F (Finger), T (Table), W (World)
        fc1_B = cp_box_table.contact_point_A.contact_force
        fc1_T = cp_box_table.contact_point_B.contact_force
        pc1_B = cp_box_table.contact_point_A.contact_position
        pc1_T = cp_box_table.contact_point_B.contact_position

        fc2_B = cp_box_finger.contact_point_A.contact_force
        fc2_F = cp_box_finger.contact_point_B.contact_force
        pc2_B = cp_box_finger.contact_point_A.contact_position
        pc2_F = cp_box_finger.contact_point_B.contact_position

        fc1_Bs.append(fc1_B)
        fc1_Ts.append(fc1_T)
        pc1_Bs.append(pc1_B)
        pc1_Ts.append(pc1_T)
        fc2_Bs.append(fc2_B)
        fc2_Fs.append(fc2_F)
        pc2_Bs.append(pc2_B)
        pc2_Fs.append(pc2_F)

        # Constant variables
        fg_W = np.array([0, -BOX_MASS * GRAV_ACC]).reshape((-1, 1))

        # TODO maybe these should become part of the other objects?
        cos_th = sym.Variable("cos")
        sin_th = sym.Variable("sin")
        R_TB = np.array([[cos_th, -sin_th], [sin_th, cos_th]])
        R_WB = R_TB  # World frame is the same as table frame

        prog.AddDecisionVariables(np.array([cos_th, sin_th]))
        prog.AddDecisionVariables(cp_box_table.variables)
        prog.AddDecisionVariables(cp_box_finger.variables)

        if USE_FRICTION_CONE_CONSTRAINT:
            prog.AddLinearConstraint(
                cp_box_table.contact_point_A.create_friction_cone_constraints()
            )
            prog.AddLinearConstraint(
                cp_box_table.contact_point_B.create_friction_cone_constraints()
            )
            prog.AddLinearConstraint(
                cp_box_finger.contact_point_A.create_friction_cone_constraints()
            )

        if USE_FORCE_BALANCE_CONSTRAINT:
            sum_of_forces_B = fc1_B + fc2_B + R_WB.T.dot(fg_W)
            prog.AddLinearConstraint(eq(sum_of_forces_B, 0))

        if USE_MOMENT_BALANCE:
            # Add moment balance using a linear relaxation
            sum_of_moments_B = cross_2d(pc1_B, fc1_B) + cross_2d(pc2_B, fc2_B)

            MAX_FORCE = 4
            # This is hardcoded for now and should be generalized. It is used to define the mccormick envelopes
            variable_bounds = {
                "contact_2_box_lam": (0.0, 1.0),
                "contact_2_box_c_n": (0, MAX_FORCE),
            }
            (
                relaxed_sum_of_moments,
                new_vars,
                mccormick_envelope_constraints,
            ) = relax_bilinear_expression(sum_of_moments_B, variable_bounds)
            prog.AddDecisionVariables(new_vars)
            prog.AddLinearConstraint(relaxed_sum_of_moments == 0)
            for c in mccormick_envelope_constraints:
                prog.AddLinearConstraint(c)

        if USE_SO2_CONSTRAINT:
            # Relaxed SO(2) constraint
            prog.AddLorentzConeConstraint(1, (R_WB.T.dot(R_WB))[0, 0])

        if USE_NON_PENETRATION_CONSTRAINT:
            # Add nonpenetration constraint in table frame
            table_a_T = table.a1[0]
            pm1_B = box.p1
            pm3_B = box.p3

            nonpen_1_T = table_a_T.T.dot(R_TB).dot(pm1_B - pc1_B)[0, 0] >= 0
            nonpen_2_T = table_a_T.T.dot(R_TB).dot(pm3_B - pc1_B)[0, 0] >= 0
            prog.AddLinearConstraint(nonpen_1_T)
            prog.AddLinearConstraint(nonpen_2_T)

        if USE_SO2_CUT:
            # Add SO(2) tightening
            cut_p1 = np.array([0, 1]).reshape((-1, 1))
            cut_p2 = np.array([1, 0]).reshape((-1, 1))
            a_cut, b_cut = construct_2d_plane_from_points(cut_p1, cut_p2)
            temp = R_TB[:, 0:1]
            cut = (a_cut.T.dot(temp) - b_cut)[0, 0] >= 0
            prog.AddLinearConstraint(cut)

        if USE_QUADRATIC_COST:
            # Quadratic cost on force
            forces_squared = fc1_B.T.dot(fc1_B)[0, 0] + fc2_B.T.dot(fc2_B)[0, 0]
            prog.AddQuadraticCost(forces_squared)

        else:
            z = sym.Variable("z")
            prog.AddDecisionVariables(np.array([z]))
            prog.AddLinearCost(1 * z)

            for f in fc1_B.flatten():
                prog.AddLinearConstraint(z >= f)
                prog.AddLinearConstraint(z >= -f)

            for f in fc2_B.flatten():
                prog.AddLinearConstraint(z >= f)
                prog.AddLinearConstraint(z >= -f)

        cos_ths.append(cos_th)
        sin_ths.append(sin_th)

    # Initial and final condition
    th_initial = 0
    prog.AddLinearConstraint(cos_ths[0] == np.cos(th_initial))
    prog.AddLinearConstraint(sin_ths[0] == np.sin(th_initial))

    th_final = 0.9
    prog.AddLinearConstraint(cos_ths[-1] == np.cos(th_final))
    prog.AddLinearConstraint(sin_ths[-1] == np.sin(th_final))

    # Don't allow contact position to change
    prog.AddLinearConstraint(eq(pc2_Bs[0], pc2_Bs[1]))
    prog.AddLinearConstraint(eq(pc2_Bs[1], pc2_Bs[2]))

    # Solve
    result = Solve(prog)
    assert result.is_success()

    print(f"Cost: {result.get_optimal_cost()}")

    cos_th_vals = np.hstack([result.GetSolution(c) for c in cos_ths]).reshape((1, -1))
    sin_th_vals = np.hstack([result.GetSolution(s) for s in sin_ths]).reshape((1, -1))
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

    pm4_W = np.array([0, TABLE_HEIGHT / 2]).reshape((-1, 1))
    pc1_B = pc1_B_curve[
        0:1, :
    ].T  # NOTE: This stays fixed for the entire interval, due to being a corner! This is a quick fix
    p_WB_curve = [pm4_W - R.dot(pc1_B) for R in R_curve]

    ## Plotting

    import tkinter as tk
    from tkinter import Canvas

    app = tk.Tk()
    app.title("Box")

    canvas = Canvas(app, width=500, height=500)
    canvas.pack()

    PLOT_CENTER = np.array([200, 300]).reshape((-1, 1))
    PLOT_SCALE = 50
    FORCE_SCALE = 0.2

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

    def _plot_force(
        pos_B: npt.NDArray[np.float64],
        force_B: npt.NDArray[np.float64],
        R_WB: npt.NDArray[np.float64],
        p_WB: npt.NDArray[np.float64],
        canvas,
        color: str = "#0f0"
    ):
        force_W = np.hstack(
            [p_WB + R_WB.dot(pos_B), p_WB + R_WB.dot(pos_B + force_B)]
        )
        force_points = _make_plotable(force_W)
        canvas.create_line(force_points, width=2, arrow=tk.LAST, fill=color)

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

        sum_of_forces = fc1_B_val + fc2_B_val + R_WB_val.T.dot(f_Wg_val)
        if any(np.abs(sum_of_forces) > 1e-8):
            breakpoint()

        points_box = _make_plotable(p_WB + R_WB_val.dot(box.corners))
        points_table = _make_plotable(table.corners)

        canvas.create_polygon(points_box, fill="#88f")
        canvas.create_polygon(points_table, fill="#2f2f2f")

        _plot_force(pc1_B_val, fc1_B_val, R_WB_val, p_WB, canvas, "#0f0")
        _plot_force(pc2_B_val, fc2_B_val, R_WB_val, p_WB, canvas, "#0f0")

        _plot_force(pc1_T_val, fc1_T_val, R_WB_val.T, p_WB, canvas, "#ff3") # TODO these forces may be wrongly plotted
        _plot_force(pc2_F_val, fc2_F_val, R_WB_val.T, p_WB, canvas, "#ff3")

        grav_force = _make_plotable(np.hstack([p_WB, p_WB + f_Wg_val * det_R]))
        canvas.create_line(grav_force, width=2, arrow=tk.LAST, fill="#f00")

        canvas.update()
        time.sleep(0.05)
        if idx == 0:
            time.sleep(2)

    app.mainloop()


if __name__ == "__main__":
    plan_box_flip_up_newtons_third_law()
