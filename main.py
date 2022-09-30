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
from geometry.contact import (
    ContactMode,
    CollisionGeometry,
    CollisionPair,
    create_force_balance_constraint,
    create_possible_mode_combinations,
)
from planning.gcs import GcsPlanner, GcsContactPlanner
from visualize.visualize import animate_1d_box, plot_1d_box_positions


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
    dim = 2
    box_width = 2
    box_height = 1
    finger = CollisionGeometry("finger", dim=dim, order=2)
    table = CollisionGeometry("table", dim=dim, order=2)
    box = CollisionGeometry("box", dim=dim, order=2)

    n = np.array([[1], [0]])
    d1 = np.array([[0], [1]])
    d2 = np.array([[0], [-1]])
    d = np.hstack((d1, d2))
    # TODO only local jacobians
    # v_rel = [v_x_rel  = [v_f_x - v_b_x
    #          v_y_rel]    v_f_y - v_b_y]
    # v = [v_f_x, v_f_y, v_b_x, v_b_y]
    J_c = np.array([[1, 0, -1, 0], [0, 1, 0, -1]])
    sdf = (finger.pos + box_width - box.pos).x[0:1, :]

    cp1 = CollisionPair(
        finger,
        box,
        friction_coeff=0.5,
        signed_distance_func=sdf,
        normal_vector=n,
        friction_cone_rays=d,
        contact_jacobian=J_c,
    )

    # TODO this should be automatic
    n = np.array([[0], [1]])
    d1 = np.array([[1], [0]])
    d2 = np.array([[-1], [0]])
    d = np.hstack((d1, d2))
    J_c = np.array([[1, 0, -1, 0], [0, 1, 0, -1]])

    cp2 = CollisionPair(
        box,
        table,
        friction_coeff=0.5,
        signed_distance_func=sdf,
        normal_vector=n,
        friction_cone_rays=d,
        contact_jacobian=J_c,
    )
    # TODO normally I would need even one more collision geometry here!

    # Add force balance, clean up
    all_collision_pairs = [cp1, cp2]
    # TODO generalize, hardcoded a bit for now
    gravitational_force = np.array([[0], [-9.81 * 1], [0], [9.81 * 1]])
    force_balance_constraint = create_force_balance_constraint(
        all_collision_pairs, gravitational_force
    )

    for cp in all_collision_pairs:
        cp.add_constraint_to_modes(force_balance_constraint)

    slack_vars = np.concatenate([cp.slack_vars for cp in all_collision_pairs])
    force_vars = np.concatenate([cp.force_vars for cp in all_collision_pairs])

    all_vars = np.concatenate(
        [
            finger.pos.x.flatten(),
            box.pos.x.flatten(),
            table.pos.x.flatten(),
            force_vars,
            slack_vars,
        ]
    )

    for cp in all_collision_pairs:
        cp.create_mode_polyhedrons(all_vars)

    possible_contact_permutations = create_possible_mode_combinations(
        all_collision_pairs
    )
    convex_sets = [
        m1.polyhedron.Intersection(m2.polyhedron)
        for m1, m2 in possible_contact_permutations
    ]

    breakpoint()

    # TODO delete this stuff

    lam_n = BezierVariable(dim=1, order=2, name="lambda_n")
    lam_f = BezierVariable(dim=1, order=2, name="lambda_f")
    x_a = BezierVariable(dim=1, order=2, name="x_a")
    x_u = BezierVariable(dim=1, order=2, name="x_u")

    contact_jacobian = np.array([[-1, 1]])
    normal_jacobian = contact_jacobian
    tangential_jacobian = np.array([[0, -1]])  # note: hand written for this case

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

    # TODO clean up this mess
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

    plot_1d_box_positions(x_a_curves, x_u_curves)
    animate_1d_box(
        x_a_curve_values, x_u_curve_values, lambda_f_curve_values, lambda_f_curve_values
    )


def main():
    test_planning_through_contact()
    return 0


if __name__ == "__main__":
    main()
