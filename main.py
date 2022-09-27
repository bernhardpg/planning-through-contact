import matplotlib.pyplot as plt
from matplotlib import animation
import cdd

from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from typing import List, Literal

import math
from pydrake.math import le, ge, eq
import pydrake.symbolic as sym
import pydrake.geometry.optimization as opt

from pydrake.geometry.optimization import GraphOfConvexSets
from pydrake.solvers import MathematicalProgram, Solve, MathematicalProgramResult

from geometry.polyhedron import Polyhedron
from geometry.bezier import BezierCurve, BezierVariable
from geometry.contact import ContactMode
from planning.gcs import GcsPlanner, GcsContactPlanner


def create_test_polyhedrons() -> List[Polyhedron]:
    vertices = [
        np.array([[0, 0], [0, 2], [1.5, 3], [2, 0]]),
        np.array([[1, 2], [1, 4.5], [3, 4.5], [4, 1]]),
        np.array([[3, 2], [3, 3], [5, 3], [5, 2]]),
        np.array([[3, 4], [3, 5], [6, 5], [6, 4]]),
        np.array([[5, 4], [7, 6], [8, 2.5], [4.5, 2.5]]),
        # For some reason, this violates continuity
        # np.array([[4, 4], [7, 6], [8, 2.5], [4.5, 2.5]]),
    ]
    polyhedrons = [Polyhedron.from_vertices(v) for v in vertices]

    return polyhedrons


def test_bezier_curve() -> None:
    order = 2
    dim = 2

    poly = create_test_polyhedron_1()
    vertices = poly.get_vertices()
    plt.fill(vertices[:, 0], vertices[:, 1], alpha=0.3)

    x0 = np.array([0, 0.5]).reshape((-1, 1))
    xf = np.array([4, 3]).reshape((-1, 1))

    bezier_curve = BezierCurveMathProgram(order, dim)
    bezier_curve.constrain_to_polyhedron(poly)
    bezier_curve.constrain_start_pos(x0)
    bezier_curve.constrain_end_pos(xf)
    bezier_curve.calc_ctrl_points()
    path = np.concatenate(
        [bezier_curve.eval(s) for s in np.arange(0.0, 1.01, 0.01)], axis=1
    ).T

    plt.plot(path[:, 0], path[:, 1])
    plt.scatter(x0[0], x0[1])
    plt.scatter(xf[0], xf[1])

    plt.show()


def plot_polyhedrons(polys: List[Polyhedron]) -> None:
    for poly in polys:
        vertices = poly.get_vertices()
        plt.fill(vertices[:, 0], vertices[:, 1], alpha=0.3)
    plt.show()


def test_gcs() -> None:
    order = 2
    dim = 2

    polys = create_test_polyhedrons()

    path = GcsPlanner(order, polys)

    x0 = np.array([0.5, 0.5]).reshape((-1, 1))
    xf = np.array([7.0, 5.5]).reshape((-1, 1))

    breakpoint()
    v0 = path.add_point_vertex(x0, "source", "out")
    vf = path.add_point_vertex(xf, "target", "in")
    ctrl_points = path.calculate_path(v0, vf)
    curves = [
        BezierCurve.create_from_ctrl_points(dim, points) for points in ctrl_points
    ]

    # Plotting
    for poly in polys:
        vertices = poly.get_vertices()
        plt.fill(vertices[:, 0], vertices[:, 1], alpha=0.3)

    for curve in curves:
        plt.scatter(curve.ctrl_points[0, :], curve.ctrl_points[1, :])

        curve_values = np.concatenate(
            [curve.eval(s) for s in np.arange(0.0, 1.01, 0.01)], axis=1
        ).T

        plt.plot(curve_values[:, 0], curve_values[:, 1])

    plt.show()

    return


def test_planning_through_contact():
    lam_n = BezierVariable(dim=1, order=2, name="lambda_n")
    lam_f = BezierVariable(dim=1, order=2, name="lambda_f")
    x_a = BezierVariable(dim=1, order=2, name="x_a")
    x_u = BezierVariable(dim=1, order=2, name="x_u")

    friction_coeff = 0.5
    contact_jacobian = np.array([[-1, 1]])
    normal_jacobian = contact_jacobian
    tangential_jacobian = np.array([[0, -1]])

    pos_vars = np.array([x_a, x_u])
    normal_force_vars = np.array([lam_n])
    friction_force_vars = np.array([lam_f])

    l = 2.0

    no_contact_pos_constraint = x_a + l <= x_u
    no_contact = ContactMode(
        pos_vars,
        no_contact_pos_constraint,
        normal_force_vars,
        friction_force_vars,
        "no_contact",
        friction_coeff,
        normal_jacobian,
        tangential_jacobian,
    )
    contact_pos_constraint = x_a + l == x_u
    rolling_contact = ContactMode(
        pos_vars,
        contact_pos_constraint,
        normal_force_vars,
        friction_force_vars,
        "rolling_contact",
        friction_coeff,
        normal_jacobian,
        tangential_jacobian,
        name="rolling",
    )
    #    sliding_contact = ContactMode(
    #        pos_vars,
    #        contact_pos_constraint,
    #        normal_force_vars,
    #        friction_force_vars,
    #        "sliding_contact",
    #        friction_coeff,
    #        normal_jacobian,
    #        tangential_jacobian,
    #    )

    initial_position_constraints = np.concatenate([x_a == 0, x_u == 4.0])
    start = ContactMode(
        pos_vars,
        initial_position_constraints,
        normal_force_vars,
        friction_force_vars,
        "no_contact",
        friction_coeff,
        normal_jacobian,
        tangential_jacobian,
        name="start",
    )

    #    final_position_constraints = np.concatenate([x_a + l == x_u, x_u == 8.0])
    #    target = ContactMode(
    #        pos_vars,
    #        final_position_constraints,
    #        normal_force_vars,
    #        friction_force_vars,
    #        "sliding_contact",  # TODO should be sliding!
    #        friction_coeff,
    #        normal_jacobian,
    #        tangential_jacobian,
    #        name="target",
    #    )

    # modes = [start, no_contact, sliding_contact, rolling_contact, target]
    modes = [start, no_contact, rolling_contact]
    planner = GcsContactPlanner(modes)
    planner.save_graph_diagram("diagrams/graph.svg")
    planner.set_source("start")
    planner.set_target("rolling")

    ctrl_points = planner.calculate_path()
    planner.save_graph_diagram(
        "diagrams/path.svg", show_binary_edge_vars=True, use_solution=True
    )
    # TODO this is very hacky just to plot something
    ctrl_points = [ctrl_points[0], ctrl_points[1], ctrl_points[2]]
    breakpoint()

    x_a_curves = [
        BezierCurve.create_from_ctrl_points(1, points[0:3]) for points in ctrl_points
    ]
    x_u_curves = [
        BezierCurve.create_from_ctrl_points(1, points[3:6]) for points in ctrl_points
    ]
    lambda_n_curves = [
        BezierCurve.create_from_ctrl_points(1, points[6:9]) for points in ctrl_points
    ]
    lambda_f_curves = [
        BezierCurve.create_from_ctrl_points(1, points[9:12]) for points in ctrl_points
    ]
    slack_curves = [
        BezierCurve.create_from_ctrl_points(1, points[12:14]) for points in ctrl_points
    ]

    x_a_curve_values = np.concatenate(
        [
            np.concatenate(
                [x_a_curve.eval(s) for s in np.arange(0.0, 1.01, 0.01)], axis=1
            ).T
            for x_a_curve in x_a_curves
        ]
    )
    x_u_curve_values = np.concatenate(
        [
            np.concatenate(
                [x_u_curve.eval(s) for s in np.arange(0.0, 1.01, 0.01)], axis=1
            ).T
            for x_u_curve in x_u_curves
        ]
    )
    lambda_f_curve_values = np.concatenate(
        [
            np.concatenate(
                [lambda_f_curve.eval(s) for s in np.arange(0.0, 1.01, 0.01)], axis=1
            ).T
            for lambda_f_curve in lambda_f_curves
        ]
    )
    lambda_n_curve_values = np.concatenate(
        [
            np.concatenate(
                [lambda_n_curve.eval(s) for s in np.arange(0.0, 1.01, 0.01)], axis=1
            ).T
            for lambda_n_curve in lambda_n_curves
        ]
    )
    slack_curve_values = np.concatenate(
        [
            np.concatenate(
                [slack_curve.eval(s) for s in np.arange(0.0, 1.01, 0.01)], axis=1
            ).T
            for slack_curve in slack_curves
        ]
    )
    animate(
        x_a_curve_values, x_u_curve_values, lambda_f_curve_values, lambda_f_curve_values
    )

    if True:
        i = 0
        for x_a_curve, x_u_curve in zip(x_a_curves, x_u_curves):
            s_range = np.arange(0.0, 1.01, 0.01)
            # plt.scatter(curve.ctrl_points[0, :], curve.ctrl_points[1, :])
            x_a_curve_values = np.concatenate(
                [x_a_curve.eval(s) for s in s_range], axis=1
            ).T

            x_u_curve_values = np.concatenate(
                [x_u_curve.eval(s) for s in s_range], axis=1
            ).T

            plt.plot(s_range + i, x_a_curve_values, "r")
            plt.plot(s_range + i, x_u_curve_values, "b")
            i += 1

        plt.show()


def animate(x_a, x_u, lam_f, lam_n):
    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure()
    ax = plt.axes(xlim=(-4, 10), ylim=(0, 4))
    (finger,) = ax.plot([], [], "bo", lw=5)
    (box,) = ax.plot([], [], "r", lw=5)
    (force_normal,) = ax.plot([], [], "g>-", lw=2)
    (force_friction,) = ax.plot([], [], "g<-", lw=2)

    # initialization function: plot the background of each frame
    def init():
        finger.set_data([], [])
        box.set_data([], [])
        force_normal.set_data([], [])
        force_friction.set_data([], [])
        return (finger, box, force_normal, force_friction)

    # animation function.  This is called sequentially
    def animate(i):
        l = 2.0  # TODO
        height = 1.0  # TODO
        y = 1.0
        finger_height = y + 0.5
        finger.set_data(x_a[i], finger_height)

        box_com = x_u[i]
        box_shape_x = np.array(
            [box_com - l, box_com + l, box_com + l, box_com - l, box_com - l]
        )
        box_shape_y = np.array([y, y, y + height, y + height, y])
        box.set_data(box_shape_x, box_shape_y)

        force_normal.set_data([box_com - l, box_com - l + lam_n[i]], finger_height)
        force_friction.set_data([box_com - lam_f[i], box_com], y)

        return (finger, box, force_normal, force_friction)

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=400, interval=20, blit=True
    )

    plt.show()


def main():
    # test_bezier_curve()
    # test_gcs()
    # animate()
    # TODO: What about sets being unbounded??

    test_planning_through_contact()

    return 0


if __name__ == "__main__":
    main()
