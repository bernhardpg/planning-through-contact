from typing import List

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pydrake.symbolic as sym
from pydrake.math import eq
from pydrake.solvers import MathematicalProgram, Solve

from convex_relaxation.sdp import SdpRelaxation
from geometry.box import Box2d, construct_2d_plane_from_points


def cross(v1, v2):
    return (v1[0] * v2[1] - v1[1] * v2[0])[0]


def test():
    x = sym.Variable("x")
    y = sym.Variable("y")
    variables = np.array([x, y])

    prog = SdpRelaxation(variables)
    prog.add_constraint(x**2 + y**2 == 1)
    prog.add_constraint(x >= 0.5)
    prog.add_constraint(y >= 0.5)

    result = Solve(prog.prog)
    assert result.is_success()

    X_result = result.GetSolution(prog.X)
    return


def decompose_plane_through_origin(expr, vars):
    normal_vec = np.array([float(expr.coeff(var)) for var in vars])
    return normal_vec


def compute_so_2_intersection(plane_normal_vec):
    x_coeff, y_coeff = plane_normal_vec
    if y_coeff == 0:
        y_val = (1 / (1 + (y_coeff / x_coeff) ** 2)) ** 0.5
        x_val = -(y_coeff / x_coeff) * y_val
    else:
        x_val = (1 / (1 + (x_coeff / y_coeff) ** 2)) ** 0.5
        y_val = -(x_coeff / y_coeff) * x_val

    int1 = np.array([x_val, y_val]).reshape((-1, 1))
    int2 = -int1
    return int1, int2


def plot_cuts_corners_fixed(use_relaxation: bool = False):
    use_relaxation = True
    from sympy import And, Eq, plot_implicit, symbols

    BOX_WIDTH = 3
    BOX_HEIGHT = 2
    BOX_MASS = 1

    box = Box2d(BOX_WIDTH, BOX_HEIGHT)
    table = Box2d(10, 0)

    cos_th, sin_th = symbols("c s")

    # Useful variables
    u1 = np.array([cos_th, sin_th]).reshape((-1, 1))
    u2 = np.array([-sin_th, cos_th]).reshape((-1, 1))
    R_WB = np.hstack([u1, u2])

    so_2_constraint = 1 - cos_th**2 - sin_th**2

    # Fix a box corner
    p_Bm4 = box.p4
    p_Wm4 = np.array([0, 0]).reshape((-1, 1))

    # Position of COG is a function of rotation when a corner is fixed
    p_WB = p_Wm4 - R_WB.dot(p_Bm4)

    # Add Non-penetration
    p_Wm1 = R_WB.dot(box.p1) + p_WB
    p_Wm2 = R_WB.dot(box.p2) + p_WB
    p_Wm3 = R_WB.dot(box.p3) + p_WB

    a, b = table.a1
    nonpen_constr_1 = (a.T.dot(p_Wm1) - b)[0, 0]
    nonpen_constr_3 = (a.T.dot(p_Wm3) - b)[0, 0]

    # Clean up this
    a1 = decompose_plane_through_origin(nonpen_constr_1, [cos_th, sin_th])
    p1, p2 = compute_so_2_intersection(a1)
    a3 = decompose_plane_through_origin(nonpen_constr_3, [cos_th, sin_th])
    p3, p4 = compute_so_2_intersection(a3)

    planes = [a1, a3]
    points = [p1, p2, p3, p4]

    feasible_points = [p for p in points if all([a.T.dot(p) >= 0 for a in planes])]
    assert len(feasible_points) == 2

    a, b = construct_2d_plane_from_points(feasible_points[0], feasible_points[1])
    x = np.array([cos_th, sin_th]).reshape((-1, 1))
    cut = (a.T.dot(x) - b)[0, 0]

    # Unused constraints:
    # We are fixing point 4
    # nonpen_constr_4 = (a.T.dot(p_Wm4) - b)[0, 0]
    # Point2 is the corner that is the furthest away, so it won't tighten the relaxation
    # nonpen_constr_2 = (a.T.dot(p_Wm2) - b)[0, 0]

    if use_relaxation:
        plot_implicit(
            And(
                nonpen_constr_1 > 0,
                nonpen_constr_3 > 0,
                so_2_constraint > 0,
                cut > 0,
            ),
            x_var=cos_th,
            y_var=sin_th,
        )
    else:
        plot_implicit(
            And(
                nonpen_constr_1 > 0,
                nonpen_constr_3 > 0,
                Eq(so_2_constraint, 0),
                cut > 0,
            ),
            x_var=cos_th,
            y_var=sin_th,
        )


def plot_cuts_with_fixed_position(use_relaxation: bool = False):
    from sympy import And, Eq, plot_implicit, symbols

    BOX_WIDTH = 3
    BOX_HEIGHT = 2
    BOX_MASS = 1

    box = Box2d(BOX_WIDTH, BOX_HEIGHT)
    table = Box2d(10, 0)

    cos_th, sin_th = symbols("c s")

    # Useful variables
    p_WB = np.array([0, 1.5]).reshape((-1, 1))
    u1 = np.array([cos_th, sin_th]).reshape((-1, 1))
    u2 = np.array([-sin_th, cos_th]).reshape((-1, 1))
    R_WB = np.hstack([u1, u2])

    so_2_constraint = 1 - cos_th**2 - sin_th**2

    # Add Non-penetration
    p_Wm1 = R_WB.dot(box.p1) + p_WB
    p_Wm2 = R_WB.dot(box.p2) + p_WB
    p_Wm3 = R_WB.dot(box.p3) + p_WB
    p_Wm4 = R_WB.dot(box.p4) + p_WB

    a, b = table.a1
    nonpen_constr_1 = (a.T.dot(p_Wm1) - b)[0, 0]
    nonpen_constr_2 = (a.T.dot(p_Wm2) - b)[0, 0]
    nonpen_constr_3 = (a.T.dot(p_Wm3) - b)[0, 0]
    nonpen_constr_4 = (a.T.dot(p_Wm4) - b)[0, 0]

    if use_relaxation:
        plot_implicit(
            And(
                nonpen_constr_1 > 0,
                nonpen_constr_2 > 0,
                nonpen_constr_3 > 0,
                nonpen_constr_4 > 0,
                so_2_constraint > 0,
            ),
            x_var=cos_th,
            y_var=sin_th,
        )
    else:
        plot_implicit(
            And(
                nonpen_constr_1 > 0,
                nonpen_constr_2 > 0,
                nonpen_constr_3 > 0,
                nonpen_constr_4 > 0,
                Eq(so_2_constraint, 0),
            ),
            x_var=cos_th,
            y_var=sin_th,
        )

    # Plot box
    corners_box = np.hstack([box.p1, box.p2, box.p3, box.p4, box.p1]) + p_WB
    corners_table = np.hstack([table.p1, table.p2, table.p3, table.p4, table.p1])
    plt.plot(corners_box[0, :], corners_box[1, :])
    plt.plot(corners_table[0, :], corners_table[1, :])
    plt.ylim(-5, 5)
    ax = plt.gca()


def test_cuts():
    BOX_WIDTH = 3
    BOX_HEIGHT = 2
    BOX_MASS = 1
    GRAV_ACC = 9.81

    box = Box2d(BOX_WIDTH, BOX_HEIGHT)
    box_table = Box2d(10, 0)

    u_11 = sym.Variable("u_11")
    u_12 = sym.Variable("u_12")
    u_21 = sym.Variable("u_21")
    u_22 = sym.Variable("u_22")

    p_WB_x = sym.Variable("p_WB_x")
    p_WB_y = sym.Variable("p_WB_y")

    # Create SDP relaxation
    variables = np.array([u_11, u_12, u_21, u_22, p_WB_x, p_WB_y])
    prog = SdpRelaxation(variables)

    # Useful variables
    p_WB = np.array([p_WB_x, p_WB_y]).reshape((-1, 1))
    u1 = np.array([u_11, u_12]).reshape((-1, 1))
    u2 = np.array([u_21, u_22]).reshape((-1, 1))
    R_WB = np.hstack([u1, u2])

    # Constrain position
    prog.add_constraint(p_WB_x == 0.0)
    prog.add_constraint(p_WB_y == 1.5)

    # SO(2) constraints
    prog.add_constraint(u1.T.dot(u1)[0, 0] == 1)
    prog.add_constraint(u2.T.dot(u2)[0, 0] == 1)
    prog.add_constraint(u1.T.dot(u2)[0, 0] == 0)

    # Add Non-penetration
    p_Wm1 = R_WB.dot(box.p1) + p_WB
    p_Wm2 = R_WB.dot(box.p2) + p_WB
    p_Wm3 = R_WB.dot(box.p3) + p_WB
    p_Wm4 = R_WB.dot(box.p4) + p_WB

    a, b = box_table.a1
    nonpen_constr_1 = (a.T.dot(p_Wm1) - b)[0, 0]
    nonpen_constr_2 = (a.T.dot(p_Wm2) - b)[0, 0]
    nonpen_constr_3 = (a.T.dot(p_Wm3) - b)[0, 0]
    nonpen_constr_4 = (a.T.dot(p_Wm4) - b)[0, 0]

    prog.add_constraint(nonpen_constr_1 >= 0)
    prog.add_constraint(nonpen_constr_2 >= 0)
    prog.add_constraint(nonpen_constr_3 >= 0)
    prog.add_constraint(nonpen_constr_4 >= 0)

    # Add cuts
    prog.add_constraint(nonpen_constr_1 * nonpen_constr_1 >= 0)
    prog.add_constraint(nonpen_constr_1 * nonpen_constr_2 >= 0)
    prog.add_constraint(nonpen_constr_1 * nonpen_constr_3 >= 0)
    prog.add_constraint(nonpen_constr_1 * nonpen_constr_4 >= 0)

    prog.add_constraint(nonpen_constr_2 * nonpen_constr_1 >= 0)
    prog.add_constraint(nonpen_constr_2 * nonpen_constr_2 >= 0)
    prog.add_constraint(nonpen_constr_2 * nonpen_constr_3 >= 0)
    prog.add_constraint(nonpen_constr_2 * nonpen_constr_4 >= 0)

    prog.add_constraint(nonpen_constr_3 * nonpen_constr_1 >= 0)
    prog.add_constraint(nonpen_constr_3 * nonpen_constr_2 >= 0)
    prog.add_constraint(nonpen_constr_3 * nonpen_constr_3 >= 0)
    prog.add_constraint(nonpen_constr_3 * nonpen_constr_4 >= 0)

    prog.add_constraint(nonpen_constr_4 * nonpen_constr_1 >= 0)
    prog.add_constraint(nonpen_constr_4 * nonpen_constr_2 >= 0)
    prog.add_constraint(nonpen_constr_4 * nonpen_constr_3 >= 0)
    prog.add_constraint(nonpen_constr_4 * nonpen_constr_4 >= 0)

    rounded_solution, X_solution = prog.get_solution()
    breakpoint()


def sdp_relaxation():
    BOX_WIDTH = 3
    BOX_HEIGHT = 2
    BOX_MASS = 1
    GRAV_ACC = 9.81

    box = Box2d(BOX_WIDTH, BOX_HEIGHT)

    c_th = sym.Variable("c_th")
    s_th = sym.Variable("s_th")
    p_WB_x = sym.Variable("p_WB_x")
    p_WB_y = sym.Variable("p_WB_y")
    variables = np.array([c_th, s_th, p_WB_x, p_WB_y])

    prog = SdpRelaxation(variables)

    # SO(2) constraints
    so_2_constraint = c_th**2 + s_th**2 == 1
    prog.add_constraint(so_2_constraint)

    # Add Non-penetration
    p_WB = np.array([p_WB_x, p_WB_y]).reshape((-1, 1))
    R_WB = np.array([[c_th, -s_th], [s_th, c_th]])

    # Table hyperplane
    a = np.array([0, 1]).reshape((-1, 1))
    b = -2

    for p_Bmi in [box.p1, box.p2, box.p3, box.p4]:
        p_Wmi = p_WB + R_WB.dot(p_Bmi)
        non_pen_constraint = (a.T.dot(p_Wmi))[0, 0] >= b
        prog.add_constraint(non_pen_constraint)

    for c in eq(p_WB, 0):
        prog.add_constraint(c[0])
    solution = prog.get_solution()
    breakpoint()

    return


def simple_rotations_test(use_sdp_relaxation: bool = True):
    N_DIMS = 2
    NUM_CTRL_POINTS = 3

    prog = MathematicalProgram()

    BOX_WIDTH = 3
    BOX_HEIGHT = 2
    BOX_MASS = 1
    GRAV_ACC = 9.81

    FINGER_POS = np.array([[-BOX_WIDTH / 2], [BOX_HEIGHT / 2]])
    GROUND_CONTACT_POS = np.array([[BOX_WIDTH / 2], [-BOX_HEIGHT / 2]])

    f_gravity = np.array([[0], [-BOX_MASS * GRAV_ACC]])
    f_finger = prog.NewContinuousVariables(N_DIMS, NUM_CTRL_POINTS, "f_finger")
    f_contact = prog.NewContinuousVariables(N_DIMS, NUM_CTRL_POINTS, "f_contact")
    cos_th = prog.NewContinuousVariables(1, NUM_CTRL_POINTS, "cos_th")
    sin_th = prog.NewContinuousVariables(1, NUM_CTRL_POINTS, "sin_th")

    # Force and moment balance
    R_f_gravity = np.concatenate(
        (
            cos_th * f_gravity[0] - sin_th * f_gravity[1],
            sin_th * f_gravity[0] + cos_th * f_gravity[1],
        )
    )
    force_balance = eq(f_finger + f_contact + R_f_gravity, 0)
    moment_balance = eq(
        cross(FINGER_POS, f_finger) + cross(GROUND_CONTACT_POS, f_contact), 0
    )

    prog.AddLinearConstraint(force_balance)
    prog.AddLinearConstraint(moment_balance)

    # Force minimization cost
    prog.AddQuadraticCost(
        np.eye(N_DIMS * NUM_CTRL_POINTS),
        np.zeros((N_DIMS * NUM_CTRL_POINTS, 1)),
        f_finger.flatten(),
    )
    prog.AddQuadraticCost(
        np.eye(N_DIMS * NUM_CTRL_POINTS),
        np.zeros((N_DIMS * NUM_CTRL_POINTS, 1)),
        f_contact.flatten(),
    )

    # Path length minimization cost
    cos_cost = np.sum(np.diff(cos_th) ** 2)
    sin_cost = np.sum(np.diff(sin_th) ** 2)
    prog.AddQuadraticCost(cos_cost + sin_cost)

    # SO(2) constraint
    if use_sdp_relaxation:
        aux_vars = prog.NewContinuousVariables(3, NUM_CTRL_POINTS, "X")
        Xs = [np.array([[z[0], z[1]], [z[1], z[2]]]) for z in aux_vars.T]
        xs = [np.vstack([c, s]) for c, s in zip(cos_th.T, sin_th.T)]
        Ms = [np.block([[1, x.T], [x, X]]) for X, x in zip(Xs, xs)]
        for X, M in zip(Xs, Ms):
            prog.AddLinearConstraint(X[0, 0] + X[1, 1] - 1 == 0)
            prog.AddPositiveSemidefiniteConstraint(M)
    else:
        cos_th_sq = (cos_th * cos_th)[0]
        sin_th_sq = (sin_th * sin_th)[0]
        prog.AddLorentzConeConstraint(1, cos_th_sq[0] + sin_th_sq[0])
        prog.AddLorentzConeConstraint(1, cos_th_sq[1] + sin_th_sq[1])
        prog.AddLorentzConeConstraint(1, cos_th_sq[1] + sin_th_sq[1])

    # Initial and final condition
    th_initial = 0
    prog.AddLinearConstraint(cos_th[0, 0] == np.cos(th_initial))
    prog.AddLinearConstraint(sin_th[0, 0] == np.sin(th_initial))

    th_final = 0.5
    prog.AddLinearConstraint(cos_th[0, -1] == np.cos(th_final))
    prog.AddLinearConstraint(sin_th[0, -1] == np.sin(th_final))

    # Solve
    result = Solve(prog)
    assert result.is_success()

    # Rounding and projection onto SO(2)
    to_float = np.vectorize(lambda x: x.Evaluate())
    Ms_result = [to_float(result.GetSolution(M)) for M in Ms]
    ws, vs = zip(*[np.linalg.eig(M) for M in Ms_result])
    idx_highest_eigval = [np.argmax(w) for w in ws]
    vks = [v[:, idx] / v[0, idx] for idx, v in zip(idx_highest_eigval, vs)]

    vk_ps = [vk[[1, 2]] for vk in vks]
    xvs = [np.array([[vk_p[0], -vk_p[1]], [vk_p[1], vk_p[0]]]) for vk_p in vk_ps]

    Us, Ss, Vs = zip(*[np.linalg.svd(xv) for xv in xvs])
    R_hats = [
        U.dot(np.diag([1, np.linalg.det(U) * np.linalg.det(V)])).dot(V.T)
        for U, V in zip(Us, Vs)
    ]

    results_finger = result.GetSolution(f_finger)
    results_contact = result.GetSolution(f_contact)
    results_cos_th = np.array([R[0, 0] for R in R_hats])
    results_sin_th = np.array([R[1, 0] for R in R_hats])
    results_th = np.array(
        [
            [np.arccos(cos_th) for cos_th in results_cos_th],
            [np.arcsin(sin_th) for sin_th in results_sin_th],
        ]
    )

    # Plot
    fig, axs = plt.subplots(7, 1)
    axs[0].set_title("Finger force x")
    axs[1].set_title("Finger force y")
    axs[0].plot(results_finger[0, :])
    axs[1].plot(results_finger[1, :])

    axs[2].set_title("Contact force x")
    axs[3].set_title("Contact force y")
    axs[2].plot(results_contact[0, :])
    axs[3].plot(results_contact[1, :])

    axs[4].set_title("cos(th)")
    axs[5].set_title("sin(th)")
    axs[4].plot(results_cos_th)
    axs[5].plot(results_sin_th)

    axs[6].plot(results_th.T)
    axs[6].set_title("theta")

    plt.tight_layout()
    plt.show()


def plan_box_flip_up(use_sdp_relaxation: bool = True):
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
    moment_balance = cross(p_Bm4, f_Bc1) + cross(p_Bc, f_Bc2) == 0
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
