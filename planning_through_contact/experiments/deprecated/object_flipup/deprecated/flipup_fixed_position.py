import time
from dataclasses import dataclass
from typing import List, TypeVar

import numpy as np
import numpy.typing as npt
import pydrake.symbolic as sym  # type: ignore
from geometry.bezier import BezierCurve
from geometry.box import Box2d, construct_2d_plane_from_points
from geometry.utilities import cross_2d
from pydrake.math import eq
from pydrake.solvers import MathematicalProgram, Solve


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


T = TypeVar("T")


def create_rot_matrix(
    cos_ths: npt.NDArray[T],  # type: ignore
    sin_ths: npt.NDArray[T],  # type: ignore
    ctrl_point_idx: int,
) -> npt.NDArray[T]:  # type: ignore
    cos_th, sin_th = cos_ths[0, ctrl_point_idx], sin_ths[0, ctrl_point_idx]
    rot_matrix = np.array([[cos_th, -sin_th], [sin_th, cos_th]])
    return rot_matrix


# TODO: This is code for a quick experiment that should be removed long term
def plan_box_flip_up_contact_point_fixed():
    NUM_CTRL_POINTS = 3

    BOX_WIDTH = 3
    BOX_HEIGHT = 2
    box = Box2d(BOX_WIDTH, BOX_HEIGHT)

    TABLE_HEIGHT = 2
    TABLE_WIDTH = 10
    table = Box2d(TABLE_WIDTH, TABLE_HEIGHT)

    BOX_MASS = 1
    GRAV_ACC = 9.81

    prog = MathematicalProgram()

    # Constant variables
    fg_W = np.array([0, -BOX_MASS * GRAV_ACC]).reshape((-1, 1))
    pc_B = (
        0.5 * box.v2 + 0.5 * box.v3
    )  # Contact point is on the middle on the right side
    pm4_W = np.array([0, TABLE_HEIGHT / 2]).reshape((-1, 1))
    pm4_B = box.v4
    mu = 0.7

    # Decision Va0riables
    cos_ths = prog.NewContinuousVariables(1, NUM_CTRL_POINTS, "cos_th")
    sin_ths = prog.NewContinuousVariables(1, NUM_CTRL_POINTS, "sin_th")
    c_n_1s = prog.NewContinuousVariables(1, NUM_CTRL_POINTS, "c_n_1")
    c_f_1s = prog.NewContinuousVariables(1, NUM_CTRL_POINTS, "c_f_1")
    c_n_2s = prog.NewContinuousVariables(1, NUM_CTRL_POINTS, "c_n_2")
    c_f_2s = prog.NewContinuousVariables(1, NUM_CTRL_POINTS, "c_f_2")

    fixed_corner = ContactPoint(box.nc4, box.tc4, mu)
    finger_pos = ContactPoint(box.n2, box.t2, mu)

    # Add constraints to each knot point
    for ctrl_point_idx in range(0, NUM_CTRL_POINTS):
        # Convenvience variables
        c_n_1 = c_n_1s[0, ctrl_point_idx]
        c_f_1 = c_f_1s[0, ctrl_point_idx]
        c_n_2 = c_n_2s[0, ctrl_point_idx]
        c_f_2 = c_f_2s[0, ctrl_point_idx]
        R_WB = create_rot_matrix(cos_ths, sin_ths, ctrl_point_idx)

        fc1_B = fixed_corner.force_vec_from_symbols(c_n_1, c_f_1)
        fc2_B = finger_pos.force_vec_from_symbols(c_n_2, c_f_2)

        prog.AddLinearConstraint(
            fixed_corner.create_friction_cone_constraints(c_n_1, c_f_1)
        )
        prog.AddLinearConstraint(
            finger_pos.create_friction_cone_constraints(c_n_2, c_f_2)
        )

        # Force and moment balance
        force_balance = eq(fc1_B + fc2_B + R_WB.T.dot(fg_W), 0)
        moment_balance = cross_2d(pm4_B, fc1_B) + cross_2d(pc_B, fc2_B) == 0
        prog.AddLinearConstraint(force_balance)
        prog.AddLinearConstraint(moment_balance)

        # Relaxed SO(2) constraint
        prog.AddLorentzConeConstraint(1, (R_WB.T.dot(R_WB))[0, 0])

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

        # Quadratic cost on force
        forces_squared = fc1_B.T.dot(fc1_B)[0, 0] + fc2_B.T.dot(fc2_B)[0, 0]
        prog.AddQuadraticCost(forces_squared)

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
    result = Solve(prog)
    assert result.is_success()

    # Reconstruct ctrl_points
    cos_th_vals = result.GetSolution(cos_ths)
    sin_th_vals = result.GetSolution(sin_ths)

    c_n_1_vals = result.GetSolution(c_n_1s)
    c_n_2_vals = result.GetSolution(c_n_2s)
    c_f_1_vals = result.GetSolution(c_f_1s)
    c_f_2_vals = result.GetSolution(c_f_2s)

    corner_forces = fixed_corner.force_vec_from_values(c_n_1_vals, c_f_1_vals)
    finger_forces = finger_pos.force_vec_from_values(c_n_2_vals, c_f_2_vals)

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

    canvas = Canvas(app, width=500, height=500)
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

    # TODO clean up this code!
    f_Wg_val = fg_W * FORCE_SCALE
    for idx in range(len(cos_curve)):

        canvas.delete("all")
        R_WB_val = R_curve[idx]
        det_R = np.linalg.det(R_WB_val)
        p_WB = p_WB_curve[idx]

        f_Bc1_val = corner_curve[idx].reshape((-1, 1)) * FORCE_SCALE
        f_Bc2_val = finger_curve[idx].reshape((-1, 1)) * FORCE_SCALE

        force_balance = f_Bc1_val + f_Bc2_val + R_WB_val.T.dot(f_Wg_val)
        if any(np.abs(force_balance) > 1e-8):
            breakpoint()

        points_box = make_plotable(p_WB + R_WB_val.dot(box.vertices))
        points_table = make_plotable(table.vertices)

        canvas.create_polygon(points_box, fill="#88f")
        canvas.create_polygon(points_table, fill="#2f2f2f")

        force_1_points = make_plotable(
            np.hstack([pm4_W, pm4_W + R_WB_val.dot(f_Bc1_val)])
        )
        canvas.create_line(force_1_points, width=2, arrow=tk.LAST, fill="#0f0")
        force_2_points = make_plotable(
            np.hstack(
                [p_WB + R_WB_val.dot(pc_B), p_WB + R_WB_val.dot(pc_B + f_Bc2_val)]
            )
        )
        canvas.create_line(force_2_points, width=2, arrow=tk.LAST, fill="#0f0")

        grav_force = make_plotable(np.hstack([p_WB, p_WB + f_Wg_val * det_R]))
        canvas.create_line(grav_force, width=2, arrow=tk.LAST, fill="#f00")

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
    plan_box_flip_up_contact_point_fixed()
