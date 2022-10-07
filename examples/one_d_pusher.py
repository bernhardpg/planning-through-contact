from geometry.contact import CollisionGeometry
from geometry.bezier import BezierVariable
from geometry.polyhedron import PolyhedronFormulator
from geometry.bezier import BezierCurve
from visualize.visualize import animate_1d_box

from pydrake.math import le, ge, eq
from pydrake.geometry.optimization import GraphOfConvexSets
import pydrake.symbolic as sym
import pydrake.geometry.optimization as opt
from pydrake.solvers import LinearConstraint, Binding, L1NormCost, L2NormCost, Cost, PerspectiveQuadraticCost

import numpy as np
import numpy.typing as npt

import itertools

from typing import List
import pydot

import matplotlib.pyplot as plt


def evaluate_curve_from_ctrl_points(
    ctrl_points: List[npt.NDArray[np.float64]], curve_dim: int, num_curves: int
) -> npt.NDArray[np.float64]:
    bzs = [
        BezierCurve.create_from_ctrl_points(
            dim=curve_dim * num_curves, ctrl_points=points
        )
        for points in ctrl_points
    ]
    values = np.concatenate(
        [
            np.concatenate([bz.eval(s) for s in np.arange(0.0, 1.01, 0.01)], axis=1).T
            for bz in bzs
        ]
    )
    return values


def find_path_to_target(
    edges: List[GraphOfConvexSets.Edge],
    target: GraphOfConvexSets.Vertex,
    u: GraphOfConvexSets.Vertex,
) -> List[GraphOfConvexSets.Vertex]:
    current_edge = next(e for e in edges if e.u() == u)
    v = current_edge.v()
    target_reached = v == target
    if target_reached:
        return [u] + [v]
    else:
        return [u] + find_path_to_target(edges, target, v)


def plan_for_one_d_pusher():
    # Bezier curve params
    dim = 1
    order = 2

    # Physical params
    mass = 1  # kg
    g = 9.81  # m/s^2
    mg = mass * g
    l = 2
    friction_coeff = 0.5

    finger = CollisionGeometry(dim=dim, order=order, name="finger")
    box = CollisionGeometry(dim=dim, order=order, name="box")

    x_f = finger.pos.x
    v_f = finger.vel.x

    x_b = box.pos.x
    v_b = box.vel.x

    lam_n = BezierVariable(dim=dim, order=order, name="lam_n").x
    lam_f = BezierVariable(dim=dim, order=order, name="lam_f").x

    sdf = x_b - x_f - l

    # "No contact" vertex
    no_contact = []
    no_contact.append(ge(sdf, 0))
    no_contact.append(eq(lam_n, 0))
    no_contact.append(eq(v_b, 0))
    no_contact.append(eq(lam_f, -lam_n))

    # "Touching" vertex
    touching = []
    touching.append(eq(sdf, 0))
    touching.append(ge(lam_n, 0))
    touching.append(eq(v_b, 0))
    touching.append(eq(lam_f, -lam_n))

    # "Pushing right" vertex
    pushing_right = []
    pushing_right.append(eq(sdf, 0))
    pushing_right.append(ge(lam_n, 0))
    pushing_right.append(ge(v_b, 0))
    pushing_right.append(eq(lam_f, -friction_coeff * mg))
    pushing_right.append(eq(lam_f, -lam_n))

    # Create the convex sets
    all_variables = np.concatenate([var.flatten() for var in [x_f, x_b, lam_n, lam_f]])
    contact_modes = [no_contact, touching, pushing_right]
    mode_names = ["no_contact", "touching", "pushing_right"]
    polyhedrons = [
        PolyhedronFormulator(mode).formulate_polyhedron(
            variables=all_variables, make_bounded=True
        )
        for mode in contact_modes
    ]

    # Add Vertices
    gcs = GraphOfConvexSets()
    for name, poly in zip(mode_names, polyhedrons):
        gcs.AddVertex(poly, name)

    # Add edges between all vertices
    for u, v in itertools.permutations(gcs.Vertices(), 2):
        gcs.AddEdge(u, v, name=f"({u.name()},{v.name()})")

    # Add source node
    x_f_0 = 0
    x_b_0 = 4.0
    lam_n_0 = 0.0
    lam_f_0 = 0.0
    source = []
    source.append(eq(x_f, x_f_0))
    source.append(eq(x_b, x_b_0))
    source.append(eq(lam_n, lam_n_0))
    source.append(eq(lam_f, lam_f_0))
    source_polyhedron = PolyhedronFormulator(source).formulate_polyhedron(
        variables=all_variables, make_bounded=True
    )
    gcs.AddVertex(source_polyhedron, "source")

    # Add target node
    x_f_T = 0.0
    x_b_T = 8.0
    target = []
    target.append(eq(x_f, x_f_T))
    target.append(eq(x_b, x_b_T))

    target.append(ge(sdf, 0))
    target.append(eq(lam_n, 0))
    target.append(eq(v_b, 0))
    target.append(eq(lam_f, -lam_n))

    target_polyhedron = PolyhedronFormulator(target).formulate_polyhedron(
        variables=all_variables, make_bounded=True
    )
    gcs.AddVertex(target_polyhedron, "target")

    # Add no_contact node twice
    gcs.AddVertex(polyhedrons[0], "no_contact_2")

    # Connect source and target node
    vertices = {v.name(): v for v in gcs.Vertices()}
    gcs.AddEdge(vertices["pushing_right"], vertices["no_contact_2"])
    gcs.AddEdge(vertices["source"], vertices["no_contact"], "(source, no_contact)")
    gcs.AddEdge(vertices["no_contact_2"], vertices["target"])

    # Create position continuity constraints
    pos_vars = np.vstack((x_f, x_b))
    first_pos_vars = pos_vars[:, 0]
    last_pos_vars = pos_vars[:, -1]
    A_first = sym.DecomposeLinearExpressions(first_pos_vars, all_variables)
    A_last = sym.DecomposeLinearExpressions(last_pos_vars, all_variables)
    for e in gcs.Edges():
        xu, xv = e.xu(), e.xv()
        constraints = eq(A_last.dot(xu), A_first.dot(xv))
        for c in constraints:
            e.AddConstraint(c)

    # Create cost
    diffs = pos_vars[:, 1:] - pos_vars[:, :-1]
    A = sym.DecomposeLinearExpressions(diffs.flatten(), all_variables)
    b = np.zeros((A.shape[0], 1))
    path_length_cost = L2NormCost(A, b)
    for v in gcs.Vertices():
        cost = Binding[Cost](path_length_cost, v.x())
        v.AddCost(cost)

    # TODO I think I may have found another bug
    energy_cost = PerspectiveQuadraticCost(A, b)
    if False:
        for v in gcs.Vertices():
            e_cost = Binding[Cost](energy_cost, v.x())
            v.AddCost(e_cost)

    if False:
        for e in gcs.Edges():
            e_cost = Binding[Cost](energy_cost, e.xu())
            e.AddCost(e_cost)

    # Solve the problem
    options = opt.GraphOfConvexSetsOptions()
    options.convex_relaxation = False
    source = vertices["source"]
    target = vertices["target"]

    graphviz = gcs.GetGraphvizString()
    data = pydot.graph_from_dot_data(graphviz)[0]
    data.write_svg("graph.svg")

    result = gcs.SolveShortestPath(source, target, options)
    assert result.is_success()
    print("Result is success!")

    graphviz = gcs.GetGraphvizString(result, False, precision=1)
    data = pydot.graph_from_dot_data(graphviz)[0]
    data.write_svg("graph_solution.svg")

    # Retrieve path from result
    flow_variables = [e.phi() for e in gcs.Edges()]
    flow_results = [result.GetSolution(p) for p in flow_variables]
    active_edges = [
        edge for edge, flow in zip(gcs.Edges(), flow_results) if flow >= 0.99
    ]
    path = find_path_to_target(active_edges, target, source)
    vertex_values = [result.GetSolution(v.x()) for v in path]
    print("Path:")
    print([v.name() for v in path])

    # Create Bezier Curve
    curve = evaluate_curve_from_ctrl_points(vertex_values, curve_dim=dim, num_curves=4)

    plt.plot(curve)
    plt.legend(["x_box", "x_finger", "normal_force", "friction_force"])
    animate_1d_box(curve[:, 0], curve[:, 1], curve[:, 2], curve[:, 3])
    breakpoint()

    return
