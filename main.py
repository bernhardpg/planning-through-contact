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
    tangential_jacobian = -contact_jacobian

    pos_vars = np.array([x_a, x_u])
    normal_force_vars = np.array([lam_n])
    friction_force_vars = np.array([lam_f])

    l = 0.5

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
    )
    sliding_contact = ContactMode(
        pos_vars,
        contact_pos_constraint,
        normal_force_vars,
        friction_force_vars,
        "sliding_contact",
        friction_coeff,
        normal_jacobian,
        tangential_jacobian,
    )

    initial_position_constraints = np.array([x_a == 0, x_u == 4.0])
    source = ContactMode(
        pos_vars,
        initial_position_constraints,
        normal_force_vars,
        friction_force_vars,
        "no_contact",
        friction_coeff,
        normal_jacobian,
        tangential_jacobian,
        name="source",
    )

    # TODO: How to properly ensure that it cannot go from nc to target?
    # Maybe it already cant because of edge constraints?
    final_position_constraints = np.array(x_u == 8.0)
    target = ContactMode(
        pos_vars,
        final_position_constraints,
        normal_force_vars,
        friction_force_vars,
        "sliding_contact",
        friction_coeff,
        normal_jacobian,
        tangential_jacobian,
        name="target",
    )

    modes = [no_contact, rolling_contact, sliding_contact, source, target]
    planner = GcsContactPlanner(modes)

    source_vertex = next(v for v in planner.gcs.Vertices() if v.name() == "source")
    target_vertex = next(v for v in planner.gcs.Vertices() if v.name() == "target")
    ctrl_points = planner.calculate_path(source_vertex, target_vertex)
    # TODO this is very hacky just to plot something
    ctrl_points.reverse()  # TODO: remove reverse

    x_a_curves = [
        BezierCurve.create_from_ctrl_points(1, points[0:3]) for points in ctrl_points
    ]
    x_u_curves = [
        BezierCurve.create_from_ctrl_points(1, points[3:6]) for points in ctrl_points
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
    animate(x_a_curve_values, x_u_curve_values)

    #    i = 0
    #    for x_a_curve, x_u_curve in zip(x_a_curves, x_u_curves):
    #        s_range = np.arange(0.0, 1.01, 0.01)
    #        # plt.scatter(curve.ctrl_points[0, :], curve.ctrl_points[1, :])
    #        x_a_curve_values = np.concatenate(
    #            [x_a_curve.eval(s) for s in s_range], axis=1
    #        ).T
    #
    #        x_u_curve_values = np.concatenate(
    #            [x_u_curve.eval(s) for s in s_range], axis=1
    #        ).T
    #
    #        plt.plot(s_range + i, x_a_curve_values, "r")
    #        plt.plot(s_range + i, x_u_curve_values, "b")
    #        i += 1

    # plt.show()


def animate(x_a, x_u):
    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure()
    ax = plt.axes(xlim=(-4, 10), ylim=(0, 4))
    (finger,) = ax.plot([], [], "bo", lw=2)
    (box,) = ax.plot([], [], "r", lw=5)

    # initialization function: plot the background of each frame
    def init():
        finger.set_data([], [])
        box.set_data([], [])
        return (
            finger,
            box,
        )

    # animation function.  This is called sequentially
    def animate(i):
        l = 2.0  # TODO
        height = 1.0  # TODO
        y = 1.0
        finger.set_data(x_a[i], y)
        box_com = x_u[i]
        box_shape_x = np.array(
            [box_com - l, box_com + l, box_com + l, box_com - l, box_com - l]
        )
        box_shape_y = np.array([y, y, y + height, y + height, y])
        box.set_data(box_shape_x, box_shape_y)
        return (
            finger,
            box,
        )

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=200, interval=20, blit=True
    )

    plt.show()


def main():
    # test_bezier_curve()
    # test_gcs()
    # animate()
    test_planning_through_contact()

    return 0


if __name__ == "__main__":
    main()
