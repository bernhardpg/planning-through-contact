import time
from dataclasses import dataclass
from typing import List, TypeVar

import numpy as np
import numpy.typing as npt
import pydrake.symbolic as sym  # type: ignore
from convex_relaxation.mccormick import relax_bilinear_expression
from convex_relaxation.sdp import create_sdp_relaxation
from geometry.bezier import BezierCurve
from geometry.two_d.box_2d import Box2d, construct_2d_plane_from_points
from geometry.utilities import cross_2d
from pydrake.math import eq, ge
from pydrake.solvers import MathematicalProgram, Solve
from visualize.colors import COLORS

T = TypeVar("T")


def _create_rot_matrix(
    cos_ths: npt.NDArray[T],  # type: ignore
    sin_ths: npt.NDArray[T],  # type: ignore
    ctrl_point_idx: int,
) -> npt.NDArray[T]:  # type: ignore
    cos_th, sin_th = cos_ths[0, ctrl_point_idx], sin_ths[0, ctrl_point_idx]
    rot_matrix = np.array([[cos_th, -sin_th], [sin_th, cos_th]])
    return rot_matrix


@dataclass
class ContactPoint:
    normal_vec: npt.NDArray[np.float64]
    tangent_vec: npt.NDArray[np.float64]
    friction_coeff: float

    def force_vec_from_symbols(
        self, normal_force: sym.Variable, friction_force: sym.Variable
    ) -> npt.NDArray[sym.Expression]:  # type: ignore
        return normal_force * self.normal_vec + friction_force * self.tangent_vec

    def force_vec_from_values(
        self, normal_force: float, friction_force: float
    ) -> npt.NDArray[np.float64]:
        force_vec = normal_force * self.normal_vec + friction_force * self.tangent_vec
        return force_vec

    def create_friction_cone_constraints(
        self, normal_force: sym.Variable, friction_force: sym.Variable
    ) -> npt.NDArray[sym.Formula]:  # type: ignore
        upper_bound = friction_force <= self.friction_coeff * normal_force
        lower_bound = -self.friction_coeff * normal_force <= friction_force
        normal_force_positive = normal_force >= 0
        return np.vstack([upper_bound, lower_bound, normal_force_positive])


# TODO: This is code for a quick experiment that should be removed long term
def plan_box_flip_up():
    USE_MCCORMICK_RELAXATION = False
    USE_SDP_RELAXATION = True

    USE_QUADRATIC_COST = True
    USE_MOMENT_BALANCE = True
    USE_FRICTION_CONE_CONSTRAINT = True

    NUM_CTRL_POINTS = 3

    BOX_WIDTH = 3.0
    BOX_HEIGHT = 2.0
    BOX_MASS = 1.0
    box = Box2d(True, "box", BOX_MASS, BOX_WIDTH, BOX_HEIGHT)

    TABLE_HEIGHT = 2.0
    TABLE_WIDTH = 10.0
    table = Box2d(False, "table", BOX_MASS, TABLE_WIDTH, TABLE_HEIGHT)

    BOX_MASS = 1
    GRAV_ACC = 9.81

    prog = MathematicalProgram()

    # Constant variables
    fg_W = np.array([0, -BOX_MASS * GRAV_ACC]).reshape((-1, 1))
    pm4_W = np.array([0, TABLE_HEIGHT / 2]).reshape((-1, 1))
    pm4_B = box.v4
    mu = 0.7

    lam = prog.NewContinuousVariables(1, 1, "lam")[
        0, 0
    ]  # convex hull variable for contact point
    prog.AddLinearConstraint(lam <= 1)
    prog.AddLinearConstraint(0 <= lam)

    pc_B = lam * box.v2 + (1 - lam) * box.v3

    # Decision Variables
    cos_ths = prog.NewContinuousVariables(1, NUM_CTRL_POINTS, "cos_th")
    sin_ths = prog.NewContinuousVariables(1, NUM_CTRL_POINTS, "sin_th")
    c_n_1s = prog.NewContinuousVariables(1, NUM_CTRL_POINTS, "c_n_1")
    c_f_1s = prog.NewContinuousVariables(1, NUM_CTRL_POINTS, "c_f_1")
    c_n_2s = prog.NewContinuousVariables(1, NUM_CTRL_POINTS, "c_n_2")
    c_f_2s = prog.NewContinuousVariables(1, NUM_CTRL_POINTS, "c_f_2")

    prog.AddBoundingBoxConstraint(-1, 1, sin_ths)
    prog.AddBoundingBoxConstraint(-1, 1, cos_ths)

    prog.AddBoundingBoxConstraint(0, np.inf, c_n_1s)
    prog.AddBoundingBoxConstraint(0, np.inf, c_n_2s)

    if USE_MCCORMICK_RELAXATION:
        variable_bounds = {"lam": (0.0, 1.0), "c_n_2": (0, 10)}

    fixed_corner = ContactPoint(box.nc4, box.tc4, mu)
    finger_pos = ContactPoint(box.n2, box.t2, mu)

    moment_balances = []
    # Add constraints to each knot point
    for ctrl_point_idx in range(0, NUM_CTRL_POINTS):
        # Convenvience variables
        c_n_1 = c_n_1s[0, ctrl_point_idx]
        c_f_1 = c_f_1s[0, ctrl_point_idx]
        c_n_2 = c_n_2s[0, ctrl_point_idx]
        c_f_2 = c_f_2s[0, ctrl_point_idx]
        R_WB = _create_rot_matrix(cos_ths, sin_ths, ctrl_point_idx)

        fc1_B = fixed_corner.force_vec_from_symbols(c_n_1, c_f_1)
        fc2_B = finger_pos.force_vec_from_symbols(c_n_2, c_f_2)

        if USE_FRICTION_CONE_CONSTRAINT:
            prog.AddLinearConstraint(
                fixed_corner.create_friction_cone_constraints(c_n_1, c_f_1)
            )
            prog.AddLinearConstraint(
                finger_pos.create_friction_cone_constraints(c_n_2, c_f_2)
            )

        # Force balance
        sum_of_forces = fc1_B + fc2_B + R_WB.T.dot(fg_W)  # type: ignore
        prog.AddLinearConstraint(eq(sum_of_forces, 0))

        if USE_MOMENT_BALANCE:
            sum_of_moments = cross_2d(pm4_B, fc1_B) + cross_2d(pc_B, fc2_B)
            if USE_MCCORMICK_RELAXATION:
                # Add moment balance using a linear relaxation
                (
                    relaxed_sum_of_moments,
                    new_vars,
                    mccormick_envelope_constraints,
                ) = relax_bilinear_expression(
                    sum_of_moments, variable_bounds
                )  # type: ignore
                prog.AddDecisionVariables(new_vars)  # type: ignore
                prog.AddLinearConstraint(relaxed_sum_of_moments == 0)
                for c in mccormick_envelope_constraints:
                    prog.AddLinearConstraint(c)
            else:
                prog.AddConstraint(sum_of_moments == 0)

            moment_balances.append(sum_of_moments)

        if USE_MCCORMICK_RELAXATION:
            # Relaxed SO(2) constraint
            prog.AddLorentzConeConstraint(1, (R_WB.T.dot(R_WB))[0, 0])  # type: ignore
        else:
            prog.AddConstraint((R_WB.T.dot(R_WB))[0, 0] == 1)

        # Add nonpenetration constraint
        table_a = table.face_1[0]
        pm1_B = box.v1
        pm3_B = box.v3

        nonpen_1 = table_a.T.dot(R_WB).dot(pm1_B - pm4_B)[0, 0] >= 0
        nonpen_2 = table_a.T.dot(R_WB).dot(pm3_B - pm4_B)[0, 0] >= 0
        prog.AddLinearConstraint(nonpen_1)
        prog.AddLinearConstraint(nonpen_2)

        # Add SO(2) tightening
        cut_p1 = np.array([0, 1]).reshape((-1, 1))
        cut_p2 = np.array([1, 0]).reshape((-1, 1))
        a_cut, b_cut = construct_2d_plane_from_points(cut_p1, cut_p2)
        temp = R_WB[:, 0:1]
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

    # Path length minimization cost
    # cos_cost = np.sum(np.diff(cos_ths) ** 2)
    # sin_cost = np.sum(np.diff(sin_ths) ** 2)
    # prog.AddQuadraticCost(cos_cost + sin_cost)

    # Initial and final condition
    th_initial = 0
    prog.AddLinearConstraint(cos_ths[0, 0] == np.cos(th_initial))
    prog.AddLinearConstraint(sin_ths[0, 0] == np.sin(th_initial))

    th_final = 0.9
    prog.AddLinearConstraint(cos_ths[0, -1] == np.cos(th_final))
    prog.AddLinearConstraint(sin_ths[0, -1] == np.sin(th_final))

    # Solve
    if USE_MCCORMICK_RELAXATION:
        result = Solve(prog)
        assert result.is_success()

        # Reconstruct ctrl_points
        cos_th_vals = result.GetSolution(cos_ths)
        sin_th_vals = result.GetSolution(sin_ths)

        c_n_1_vals = result.GetSolution(c_n_1s)
        c_n_2_vals = result.GetSolution(c_n_2s)
        c_f_1_vals = result.GetSolution(c_f_1s)
        c_f_2_vals = result.GetSolution(c_f_2s)

        lam_val = result.GetSolution(lam)
    elif USE_SDP_RELAXATION:
        relaxed_prog, X = create_sdp_relaxation(prog)
        result = Solve(relaxed_prog)
        assert result.is_success()

        x_val = result.GetSolution(X[1:, 0])
        lam_val = x_val[0]

        cos_th_vals = x_val[1:4].reshape((-1, 3))
        sin_th_vals = x_val[4:7].reshape((-1, 3))
        c_n_1_vals = x_val[7:10].reshape((-1, 3))
        c_f_1_vals = x_val[10:13].reshape((-1, 3))
        c_n_2_vals = x_val[13:16].reshape((-1, 3))
        c_f_2_vals = x_val[16:19].reshape((-1, 3))
    else:
        result = Solve(prog)
        assert result.is_success()

        # Reconstruct ctrl_points
        cos_th_vals = result.GetSolution(cos_ths)
        sin_th_vals = result.GetSolution(sin_ths)

        c_n_1_vals = result.GetSolution(c_n_1s)
        c_n_2_vals = result.GetSolution(c_n_2s)
        c_f_1_vals = result.GetSolution(c_f_1s)
        c_f_2_vals = result.GetSolution(c_f_2s)

        lam_val = result.GetSolution(lam)

    print(f"Solver details: {result.get_solver_details()}")
    print(f"Cost: {result.get_optimal_cost()}")

    # if USE_MOMENT_BALANCE:
    # relaxation_errors = np.abs(lam_val * c_n_2_vals - aux_var_vals)
    # print("Solved with relaxation errors:")
    # # print(relaxation_errors)
    #
    # moment_balance_violations = [result.GetSolution(mb) for mb in moment_balances]
    # for idx, val in enumerate(moment_balance_violations):
    #     print(f"Violation for keypoint {idx}: {val}")

    corner_forces = fixed_corner.force_vec_from_values(c_n_1_vals, c_f_1_vals)  # type: ignore
    finger_forces = finger_pos.force_vec_from_values(c_n_2_vals, c_f_2_vals)  # type: ignore

    sin_curve = BezierCurve.create_from_ctrl_points(sin_th_vals).eval_entire_interval()
    cos_curve = BezierCurve.create_from_ctrl_points(cos_th_vals).eval_entire_interval()

    R_curve = [
        np.array([[c[0], -s[0]], [s[0], c[0]]]) for c, s in zip(cos_curve, sin_curve)
    ]

    corner_curve = BezierCurve.create_from_ctrl_points(
        corner_forces
    ).eval_entire_interval()
    finger_curve = BezierCurve.create_from_ctrl_points(
        finger_forces
    ).eval_entire_interval()

    p_WB_curve = [pm4_W - R.dot(pm4_B) for R in R_curve]

    ## Plotting

    import tkinter as tk
    from tkinter import Canvas

    app = tk.Tk()
    app.title("Box")

    canvas = Canvas(app, width=500, height=500, bg="white")
    canvas.pack()

    PLOT_CENTER = np.array([200, 300]).reshape((-1, 1))
    PLOT_SCALE = 50
    FORCE_SCALE = 0.2

    def flatten_points(points):
        return list(points.flatten(order="F"))

    def make_plotable(
        points: npt.NDArray[np.float64],
        scale: float = PLOT_SCALE,
        center: float = PLOT_CENTER,
    ) -> List[float]:
        points_flipped_y_axis = np.vstack([points[0, :], -points[1, :]])
        points_transformed = points_flipped_y_axis * scale + center
        plotable_points = flatten_points(points_transformed)
        return plotable_points

    CONTACT_COLOR = COLORS["dodgerblue4"]
    GRAVITY_COLOR = COLORS["blueviolet"]
    BOX_COLOR = COLORS["aquamarine4"]
    TABLE_COLOR = COLORS["bisque3"]
    FINGER_COLOR = COLORS["firebrick3"]

    # TODO clean up this code!
    f_Wg_val = fg_W * FORCE_SCALE
    for idx in range(len(cos_curve)):
        canvas.delete("all")
        R_WB_val = R_curve[idx]
        det_R = np.linalg.det(R_WB_val)
        p_WB = p_WB_curve[idx]

        f_Bc1_val = corner_curve[idx].reshape((-1, 1)) * FORCE_SCALE
        f_Bc2_val = finger_curve[idx].reshape((-1, 1)) * FORCE_SCALE

        # sum_of_forces = f_Bc1_val + f_Bc2_val + R_WB_val.T.dot(f_Wg_val)
        # if any(np.abs(sum_of_forces) > 1e-8):
        #     breakpoint()

        points_box = make_plotable(p_WB + R_WB_val.dot(box.vertices_for_plotting))
        points_table = make_plotable(table.vertices_for_plotting)

        canvas.create_polygon(points_box, fill=BOX_COLOR.hex_format())
        canvas.create_polygon(points_table, fill=TABLE_COLOR.hex_format())

        force_1_points = make_plotable(
            np.hstack([pm4_W, pm4_W + R_WB_val.dot(f_Bc1_val)])
        )
        canvas.create_line(
            force_1_points, width=2, arrow=tk.LAST, fill=CONTACT_COLOR.hex_format()
        )
        pc_B_val = box.v2 * lam_val + box.v3 * (1 - lam_val)
        force_2_points = make_plotable(
            np.hstack(
                [
                    p_WB + R_WB_val.dot(pc_B_val),
                    p_WB + R_WB_val.dot(pc_B_val + f_Bc2_val),
                ]
            )
        )
        canvas.create_line(
            force_2_points, width=2, arrow=tk.LAST, fill=CONTACT_COLOR.hex_format()
        )

        grav_force = make_plotable(np.hstack([p_WB, p_WB + f_Wg_val * det_R]))
        canvas.create_line(
            grav_force, width=2, arrow=tk.LAST, fill=GRAVITY_COLOR.hex_format()
        )

        canvas.update()
        time.sleep(0.05)
        if idx == 0:
            time.sleep(2)

        # a = np.array([10,10,10,100,200,100,200,10])
        # canvas.create_polygon(list(a), fill="#f32")
        #
        # b = np.array([10,10,10,100,200,100,200,10]) + 100
        # canvas.create_polygon(list(b), fill="#34f")
        #
        # c = np.array([10,10,10,100,200,100,200,10]) + 150
        # canvas.create_polygon(list(c), fill="#3f5")

        # canvas.create_line([10, 10, 10, 100, 50, 100])

    app.mainloop()


if __name__ == "__main__":
    plan_box_flip_up()
