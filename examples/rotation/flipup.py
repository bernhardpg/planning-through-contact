from typing import List

import numpy as np
import numpy.typing as npt
from pydrake.math import eq
from pydrake.solvers import MathematicalProgram, Solve

from geometry.box import Box2d, construct_2d_plane_from_points
from geometry.utilities import cross_2d


def plan_box_flip_up():

    N_DIMS = 2
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
    f_Wg = np.array([0, -BOX_MASS * GRAV_ACC]).reshape((-1, 1))
    p_Bc = (
        0.5 * box.p2 + 0.5 * box.p3
    )  # Contact point is on the middle on the right side
    p_Wm4 = np.array([0, TABLE_HEIGHT / 2]).reshape((-1, 1))
    p_Bm4 = box.p4
    mu = 0.7

    # Decision Va0riables
    c_th = prog.NewContinuousVariables(1, 1, "c_th")[0, 0]
    s_th = prog.NewContinuousVariables(1, 1, "s_th")[0, 0]
    c_n_1 = prog.NewContinuousVariables(1, 1, "c_n_1")[0, 0]
    c_f_1 = prog.NewContinuousVariables(1, 1, "c_f_1")[0, 0]
    c_n_2 = prog.NewContinuousVariables(1, 1, "c_n_2")[0, 0]
    c_f_2 = prog.NewContinuousVariables(1, 1, "c_f_2")[0, 0]

    # Convenvience variables
    R_WB = np.array([[c_th, -s_th], [s_th, c_th]])
    f_Bc1 = c_n_1 * box.nc4 + c_f_1 * box.tc4
    f_Bc2 = c_n_2 * box.n2 + c_f_2 * box.t2

    # Friction cone
    prog.AddLinearConstraint(c_f_1 <= mu * c_n_1)
    prog.AddLinearConstraint(-mu * c_n_1 <= c_f_1)
    prog.AddLinearConstraint(c_f_2 <= mu * c_n_2)
    prog.AddLinearConstraint(-mu * c_n_2 <= c_f_2)

    # Force and moment balance
    force_balance = eq(f_Bc1 + f_Bc2 + R_WB.T.dot(f_Wg), 0)
    moment_balance = cross_2d(p_Bm4, f_Bc1) + cross_2d(p_Bc, f_Bc2) == 0
    prog.AddLinearConstraint(force_balance)
    prog.AddLinearConstraint(moment_balance)

    # Relaxed SO(2) constraint
    c_th_sq = c_th**2
    s_th_sq = s_th**2
    prog.AddLorentzConeConstraint(1, c_th_sq + s_th_sq)

    # Add nonpenetration constraint
    table_a = table.a1[0]
    p_Bm1 = box.p1
    p_Bm3 = box.p3

    nonpen_1 = table_a.T.dot(R_WB).dot(p_Bm1 - p_Bm4)[0, 0] >= 0
    nonpen_2 = table_a.T.dot(R_WB).dot(p_Bm3 - p_Bm4)[0, 0] >= 0
    prog.AddLinearConstraint(nonpen_1)
    prog.AddLinearConstraint(nonpen_2)

    # Add SO(2) tightening
    cut_p1 = np.array([0, 1]).reshape((-1, 1))
    cut_p2 = np.array([1, 0]).reshape((-1, 1))
    a_cut, b_cut = construct_2d_plane_from_points(cut_p1, cut_p2)
    temp = R_WB[:, 0:1]
    cut = (a_cut.T.dot(temp) - b_cut)[0, 0] >= 0
    prog.AddLinearConstraint(cut)

    # # Force minimization cost
    # prog.AddQuadraticCost(
    #     np.eye(N_DIMS * NUM_CTRL_POINTS),
    #     np.zeros((N_DIMS * NUM_CTRL_POINTS, 1)),
    #     f_Bc1.flatten(),
    # )
    # prog.AddQuadraticCost(
    #     np.eye(N_DIMS * NUM_CTRL_POINTS),
    #     np.zeros((N_DIMS * NUM_CTRL_POINTS, 1)),
    #     f_Bc2.flatten(),
    # )

    # # Path length minimization cost
    # c_cost = np.sum(np.diff(c_th) ** 2)
    # s_cost = np.sum(np.diff(s_th) ** 2)
    # prog.AddQuadraticCost(c_cost + s_cost)
    #
    # # Initial and final condition
    # th_initial = 0
    # prog.AddLinearConstraint(cos_th[0, 0] == np.cos(th_initial))
    # prog.AddLinearConstraint(sin_th[0, 0] == np.sin(th_initial))
    #
    # th_final = 0.5
    # prog.AddLinearConstraint(cos_th[0, -1] == np.cos(th_final))
    # prog.AddLinearConstraint(sin_th[0, -1] == np.sin(th_final))

    # Solve
    result = Solve(prog)
    assert result.is_success()

    R_WB_val = result.GetSolution(R_WB)
    FORCE_SCALE = 0.2
    f_Bc1_val = result.GetSolution(f_Bc1) * FORCE_SCALE
    f_Bc2_val = result.GetSolution(f_Bc2) * FORCE_SCALE
    f_Wg_val = f_Wg * FORCE_SCALE

    p_WB = p_Wm4 - R_WB_val.dot(p_Bm4)

    import tkinter as tk
    from tkinter import Canvas

    app = tk.Tk()
    app.title("Box")

    canvas = Canvas(app, width=500, height=500)
    canvas.pack()

    PLOT_CENTER = np.array([200, 300]).reshape((-1, 1))
    PLOT_SCALE = 50

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

    points_box = make_plotable(p_WB + R_WB_val.dot(box.corners))
    points_table = make_plotable(table.corners)

    canvas.create_polygon(points_box, fill="#88f")
    canvas.create_polygon(points_table, fill="#2f2f2f")

    force_1_points = make_plotable(np.hstack([p_Wm4, p_Wm4 + R_WB_val.dot(f_Bc1_val)]))
    canvas.create_line(force_1_points, width=2, arrow=tk.LAST, fill="#0f0")
    force_2_points = make_plotable(
        np.hstack([p_WB + R_WB_val.dot(p_Bc), p_WB + R_WB_val.dot(p_Bc + f_Bc2_val)])
    )
    canvas.create_line(force_2_points, width=2, arrow=tk.LAST, fill="#0f0")

    grav_force = make_plotable(np.hstack([p_WB, p_WB + f_Wg_val]))
    canvas.create_line(grav_force, width=2, arrow=tk.LAST, fill="#0ff")

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
