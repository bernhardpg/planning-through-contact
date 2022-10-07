from geometry.bezier import BezierVariable
from geometry.polyhedron import PolyhedronFormulator
from geometry.bezier import BezierCurve
from visualize.visualize import animate_1d_box

from dataclasses import dataclass

from pydrake.math import le, ge, eq
from pydrake.geometry.optimization import GraphOfConvexSets
import pydrake.symbolic as sym
import pydrake.geometry.optimization as opt
from pydrake.solvers import (
    LinearConstraint,
    Binding,
    L1NormCost,
    L2NormCost,
    Cost,
    PerspectiveQuadraticCost,
)

import numpy as np
import numpy.typing as npt

import itertools

from typing import List
import pydot

import matplotlib.pyplot as plt


def evaluate_curve_from_ctrl_points(
    ctrl_points: List[npt.NDArray[np.float64]], curve_dim: int, num_curves: int
) -> npt.NDArray[np.float64]:
    breakpoint()
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


# TODO Plan:
# 1. Automatically enumerate contact mode combinations from hand-specified modes
# 2. Make an object for handling this
# 3. Extend to y-axis
# 4. Automatically create mode constraints
# 5. Extend with friction rays
# 6. Jacobians
# 7. Deal with multiple visits to the same node


@dataclass
class RigidBody:
    name: str
    dim: int
    order: int = 2

    def __post_init__(self) -> None:
        self.pos = BezierVariable(self.dim, self.order, name=f"{self.name}_pos")

    @property
    def vel(self) -> BezierVariable:
        return self.pos.get_derivative()


@dataclass
class ContactMode:
    name: str
    constraints: List[npt.NDArray[sym.Formula]]
    all_vars: npt.NDArray[sym.Variable]

    def __post_init__(self):
        self.polyhedron = PolyhedronFormulator(self.constraints).formulate_polyhedron(
            variables=self.all_vars, make_bounded=True
        )


@dataclass
class CollisionPair:
    body_a: RigidBody
    body_b: RigidBody
    friction_coeff: float
    sdf: BezierVariable
    order: int = 2

    @property
    def name(self) -> str:
        return f"{self.body_a.name}_{self.body_b.name}"

    def __post_init__(self):
        self.lam_n = BezierVariable(
            dim=1, order=self.order, name=f"{self.name}_lam_n"
        ).x
        self.lam_f = BezierVariable(
            dim=1, order=self.order, name=f"{self.name}_lam_f"
        ).x
        self.rel_sliding_vel = self.body_a.vel.x - self.body_b.vel.x
        self.additional_constraints = []

    def add_constraint_to_all_modes(self, constraints) -> None:
        self.additional_constraints = sum(
            [self.additional_constraints, constraints], []
        )

    def add_force_balance(self, force_balance):
        self.force_balance = force_balance

    def create_contact_modes(self, all_variables):
        assert self.force_balance is not None

        no_contact_constraints = [
            ge(self.sdf, 0),
            eq(self.lam_n, 0),
            *self.force_balance,
            *self.additional_constraints,
        ]

        rolling_constraints = [
            eq(self.sdf, 0),
            ge(self.lam_n, 0),
            le(self.lam_f, self.friction_coeff * self.lam_n),
            ge(self.lam_f, -self.friction_coeff * self.lam_n),
            *self.force_balance,
            *self.additional_constraints,
        ]

        sliding_right_constraints = [
            ge(self.rel_sliding_vel, 0),
            eq(self.lam_f, -self.friction_coeff * self.lam_n),
            *self.force_balance,
            *self.additional_constraints,
        ]

        sliding_left_constraints = [
            le(self.rel_sliding_vel, 0),
            eq(self.lam_f, self.friction_coeff * self.lam_n),
            *self.force_balance,
            *self.additional_constraints,
        ]

        modes_constraints = [
            ("no_contact", no_contact_constraints),
            ("rolling", rolling_constraints),
            ("sliding_right", sliding_right_constraints),
            ("sliding_left", sliding_left_constraints),
        ]

        self.contact_modes = [
            ContactMode(f"{self.name}_{name}", constraints, all_variables)
            for name, constraints in modes_constraints
        ]


def plan_for_one_d_pusher_2():
    # Bezier curve params
    dim = 2
    order = 2

    # Physical params
    mass = 1  # kg
    g = 9.81  # m/s^2
    mg = mass * g
    l = 2
    h = 1
    friction_coeff = 0.5

    # Define variables
    finger = RigidBody(dim=dim, order=order, name="finger")
    box = RigidBody(dim=dim, order=order, name="box")
    ground = RigidBody(dim=dim, order=order, name="ground")

    x_f = finger.pos.x[0, :]
    y_f = finger.pos.x[1, :]
    x_b = box.pos.x[0, :]
    y_b = box.pos.x[1, :]
    x_g = ground.pos.x[0, :]
    y_g = ground.pos.x[1, :]

    sdf_finger_box = x_b - x_f - l
    sdf_box_ground = y_b - y_g - h

    pair_finger_box = CollisionPair(finger, box, friction_coeff, sdf_finger_box)
    pair_box_ground = CollisionPair(box, ground, friction_coeff, sdf_box_ground)

    force_balance = [
        eq(pair_finger_box.lam_n, -pair_box_ground.lam_f),
        eq(pair_box_ground.lam_n, -pair_box_ground.lam_f + mg),
    ]
    pair_finger_box.add_force_balance(force_balance)
    pair_box_ground.add_force_balance(force_balance)

    no_ground_motion = [eq(x_g, 0), eq(y_g, 0)]
    no_y_motion = [eq(y_f, h), eq(y_b, h)]
    additional_constraints = [*no_ground_motion, *no_y_motion]
    pair_finger_box.add_constraint_to_all_modes(additional_constraints)
    pair_box_ground.add_constraint_to_all_modes(additional_constraints)

    all_variables = np.concatenate(
        [
            x_f.flatten(),
            y_f.flatten(),
            x_b.flatten(),
            y_b.flatten(),
            x_g.flatten(),
            y_g.flatten(),
            pair_finger_box.lam_n.flatten(),
            pair_finger_box.lam_f.flatten(),
            pair_box_ground.lam_n.flatten(),
            pair_box_ground.lam_f.flatten(),
        ]
    )

    pair_finger_box.create_contact_modes(all_variables)
    pair_box_ground.create_contact_modes(all_variables)

    contact_pairs = [pair_finger_box.contact_modes, pair_box_ground.contact_modes]

    possible_contact_permutations = itertools.product(*contact_pairs)
    convex_sets = {
        f"{mode_1.name}_W_{mode_2.name}": mode_1.polyhedron.Intersection(
            mode_2.polyhedron
        )
        for (mode_1, mode_2) in possible_contact_permutations
        if mode_1.polyhedron.IntersectsWith(mode_2.polyhedron)
    }

    # Add Vertices
    gcs = GraphOfConvexSets()
    for name, poly in convex_sets.items():
        gcs.AddVertex(poly, name)
    vertices = {v.name(): v for v in gcs.Vertices()}

    # Add edges between all vertices
    for u, v in itertools.permutations(gcs.Vertices(), 2):
        if u.set().IntersectsWith(v.set()):
            gcs.AddEdge(u, v)  # TODO I think that this makes sense

    # Add source node
    x_f_0 = 0
    x_b_0 = 4.0
    y_f_0 = 0.5
    y_b_0 = 0.5
    source_constraints = []
    source_constraints.append(eq(x_f, x_f_0))
    source_constraints.append(eq(x_b, x_b_0))
    source_node = vertices["finger_box_no_contact_W_box_ground_rolling"]
    source_polyhedron = (
        PolyhedronFormulator(source_constraints)
        .formulate_polyhedron(variables=all_variables, make_bounded=True)
        .Intersection(source_node.set())
    )
    source = gcs.AddVertex(source_polyhedron, "source")
    gcs.AddEdge(source, source_node)

    # Add target node
    x_f_T = 8.0 - l
    x_b_T = 8.0
    y_f_T = 0.5
    y_b_T = 0.5
    target_constraints = []
    target_constraints.append(eq(x_f, x_f_T))
    target_constraints.append(eq(x_b, x_b_T))
    target_node = vertices["finger_box_rolling_W_box_ground_sliding_right"]
    target_polyhedron = (
        PolyhedronFormulator(target_constraints)
        .formulate_polyhedron(variables=all_variables, make_bounded=True)
        .Intersection(target_node.set())
    )
    target = gcs.AddVertex(target_polyhedron, "target")
    gcs.AddEdge(target_node, target)

    # Create position continuity constraints
    pos_vars = np.vstack((x_f, y_f, x_b, y_f, x_g, y_f))
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
    if False:
        energy_cost = PerspectiveQuadraticCost(A, b)
        for v in gcs.Vertices():
            e_cost = Binding[Cost](energy_cost, v.x())
            v.AddCost(e_cost)

    if False:
        energy_cost = PerspectiveQuadraticCost(A, b)
        for e in gcs.Edges():
            e_cost = Binding[Cost](energy_cost, e.xu())
            e.AddCost(e_cost)

    # Solve the problem
    options = opt.GraphOfConvexSetsOptions()
    options.convex_relaxation = False

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

    breakpoint()
    # Create Bezier Curve
    curve = evaluate_curve_from_ctrl_points(vertex_values, curve_dim=dim, num_curves=7)

    breakpoint()
    plt.plot(curve)
    plt.legend(
        [
            "x_finger",
            "y_finger",
            "x_box",
            "y_box",
            "x_ground",
            "y_ground",
            "finger_box_normal_force",
            "finger_box_friction_force",
            "box_ground_normal_force",
            "box_ground_friction_force",
        ]
    )
    plt.show()
    animate_1d_box(
        curve[:, 0],
        curve[:, 1],
        curve[:, 2],
        curve[:, 3],
        curve[:, 4],
        curve[:, 5],
        curve[:, 6],
    )
    breakpoint()

    return


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
    if False:
        energy_cost = PerspectiveQuadraticCost(A, b)
        for v in gcs.Vertices():
            e_cost = Binding[Cost](energy_cost, v.x())
            v.AddCost(e_cost)

    if False:
        energy_cost = PerspectiveQuadraticCost(A, b)
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
    breakpoint()

    # Create Bezier Curve
    curve = evaluate_curve_from_ctrl_points(vertex_values, curve_dim=dim, num_curves=4)

    plt.plot(curve)
    plt.legend(["x_box", "x_finger", "normal_force", "friction_force"])
    animate_1d_box(curve[:, 0], curve[:, 1], curve[:, 2], curve[:, 3])
    breakpoint()

    return
