import itertools
from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pydot
import pydrake.geometry.optimization as opt
import pydrake.symbolic as sym
from pydrake.geometry.optimization import GraphOfConvexSets
from pydrake.math import eq, ge, le
from pydrake.solvers import Binding, Cost, L2NormCost, PerspectiveQuadraticCost

from geometry.bezier import BezierCurve, BezierVariable
from geometry.polyhedron import PolyhedronFormulator
from visualize.visualize import animate_2d_box


@dataclass
class VariableMetadata:
    dim: int
    order: int
    name: str

    @property
    def num_ctrl_points(self) -> int:
        return self.order + 1


def evaluate_curve_from_ctrl_points(
    ctrl_points_for_all_segments: List[npt.NDArray[np.float64]],
    metadatas: List["VariableMetadata"],
) -> npt.NDArray[np.float64]:
    def construct_bezier_curves(metadatas, ctrl_points):
        where_to_split = np.cumsum(
            [md.dim * md.num_ctrl_points for md in metadatas[:-1]]
        )
        ctrl_points_for_each_curve = np.split(ctrl_points, where_to_split)
        bzs = [
            BezierCurve.create_from_ctrl_points(dim=md.dim, ctrl_points=points)
            for md, points in zip(metadatas, ctrl_points_for_each_curve)
        ]
        return bzs

    bzs = [
        construct_bezier_curves(metadatas, ctrl_points)
        for ctrl_points in ctrl_points_for_all_segments
    ]

    values = [
        {
            md.name: np.concatenate(
                [bz.eval(s) for s in np.arange(0.0, 1.01, 0.01)], axis=1
            ).T
            for md, bz in zip(metadatas, segment)
        }
        for segment in bzs
    ]

    values_merged = {
        md.name: np.concatenate([value[md.name] for value in values])
        for md in metadatas
    }

    return values_merged


def create_intersecting_set(
    constraints: List[sym.Formula],
    all_vars: List[sym.Variable],
    vertices: GraphOfConvexSets.Vertices,
):
    constraints_as_poly = PolyhedronFormulator(constraints).formulate_polyhedron(
        variables=all_vars, make_bounded=True
    )

    vertex = next(v for v in vertices if v.set().IntersectsWith(constraints_as_poly))
    intersecting_set = constraints_as_poly.Intersection(vertex.set())
    return intersecting_set, vertex


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
# DONE:
# 1. Automatically enumerate contact mode combinations from hand-specified modes
# 2. Make an object for handling this
# 3. Extend to y-axis
# 4. Automatically create mode constraints

# Plan going forward:
# 5. Jacobians, normal_vec, friction_vec
# 6. Code cleanup
# 7. Functionality for adding source and target constraints in a nice way
# 8. Deal with multiple visits to the same node
# 9. Two fingers picking up box
# 10. Fix energy cost, should be easy!
# 11. For 3D: extend with friction rays at some points


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
    sdf: sym.Expression
    rel_tangential_sliding_vel: sym.Expression
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
            le(self.lam_f, self.friction_coeff * self.lam_n),
            ge(self.lam_f, -self.friction_coeff * self.lam_n),
            *self.force_balance,
            *self.additional_constraints,
        ]

        rolling_constraints = [
            eq(self.sdf, 0),
            ge(self.lam_n, 0),
            eq(self.rel_tangential_sliding_vel, 0),
            le(self.lam_f, self.friction_coeff * self.lam_n),
            ge(self.lam_f, -self.friction_coeff * self.lam_n),
            *self.force_balance,
            *self.additional_constraints,
        ]

        sliding_right_constraints = [
            eq(self.sdf, 0),
            ge(self.lam_n, 0),
            ge(self.rel_tangential_sliding_vel, 0),
            eq(self.lam_f, -self.friction_coeff * self.lam_n),
            *self.force_balance,
            *self.additional_constraints,
        ]

        sliding_left_constraints = [
            eq(self.sdf, 0),
            ge(self.lam_n, 0),
            le(self.rel_tangential_sliding_vel, 0),
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
    box_width = 2
    box_height = 1
    friction_coeff = 0.5

    # Define variables
    finger = RigidBody(dim=dim, order=order, name="finger")
    box = RigidBody(dim=dim, order=order, name="box")
    ground = RigidBody(dim=dim, order=order, name="ground")

    x_f = finger.pos.x[0, :]
    y_f = finger.pos.x[1, :]
    vy_f = finger.vel.x[1, :]
    x_b = box.pos.x[0, :]
    y_b = box.pos.x[1, :]
    vx_b = box.vel.x[0, :]
    vy_b = box.vel.x[1, :]
    x_g = ground.pos.x[0, :]
    y_g = ground.pos.x[1, :]
    vx_g = ground.vel.x[0, :]

    # NOTE this is the stuff the jacobians will replace
    sdf_finger_box = x_b - x_f - box_width
    sdf_box_ground = y_b - y_g - box_height

    # NOTE this is the stuff the jacobians will replace
    finger_box_rel_tangential_sliding_vel = vy_f - vy_b
    box_ground_rel_tangential_sliding_vel = vx_b - vx_g

    pair_finger_box = CollisionPair(
        finger,
        box,
        friction_coeff,
        sdf_finger_box,
        finger_box_rel_tangential_sliding_vel,
    )
    pair_box_ground = CollisionPair(
        box,
        ground,
        friction_coeff,
        sdf_box_ground,
        box_ground_rel_tangential_sliding_vel,
    )

    force_balance = [
        eq(pair_finger_box.lam_n, -pair_box_ground.lam_f),
        eq(pair_box_ground.lam_n, -pair_finger_box.lam_f + mg),
    ]
    pair_finger_box.add_force_balance(force_balance)
    pair_box_ground.add_force_balance(force_balance)

    no_ground_motion = [eq(x_g, 0), eq(y_g, 0)]
    no_box_y_motion = eq(y_b, box_height)
    finger_pos_below_box_height = le(y_f, y_b + box_height)
    additional_constraints_finger_box = [*no_ground_motion, finger_pos_below_box_height]
    additional_constraints_box_ground = [
        *no_ground_motion,
        no_box_y_motion,
        eq(pair_box_ground.lam_n, mg),
    ]
    pair_finger_box.add_constraint_to_all_modes(additional_constraints_finger_box)
    pair_box_ground.add_constraint_to_all_modes(additional_constraints_box_ground)

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

    # Add edges between all vertices
    for u, v in itertools.permutations(gcs.Vertices(), 2):
        if u.set().IntersectsWith(v.set()):
            gcs.AddEdge(u, v)  # TODO I think that this makes sense

    # Add source node
    # TODO refactor into a function
    source_constraints = [eq(x_f, 0), eq(y_f, 0.6), eq(x_b, 4.0)]
    source_set, matching_vertex = create_intersecting_set(
        source_constraints, all_variables, gcs.Vertices()
    )
    source = gcs.AddVertex(source_set, "source")
    gcs.AddEdge(source, matching_vertex)

    # Add target node
    target_constraints = [eq(x_f, 0.0), eq(x_b, 8.0)]
    target_set, matching_vertex = create_intersecting_set(
        target_constraints, all_variables, gcs.Vertices()
    )
    target = gcs.AddVertex(target_set, "target")
    gcs.AddEdge(matching_vertex, target)

    # Allow repeated visits to the same node
    # TODO: This should be sped up, as it will scale poorly
    NUM_ALLOWED_REVISITS = 1
    # Runtime: O(v * E), E ~= v^2, O(V^3)
    new_edges = []
    for i in range(NUM_ALLOWED_REVISITS):
        for v in gcs.Vertices():
            v_new = gcs.AddVertex(v.set(), f"{v.name()}_2")
            for e in gcs.Edges():
                if v == e.v():
                    new_edges.append((e.u(), v_new))
                elif v == e.u():
                    new_edges.append((v_new, e.v()))

    for u, v in new_edges:
        gcs.AddEdge(u, v)

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
    options.preprocessing = True
    options.max_rounded_paths = 10

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

    # TODO should be created automatically
    var_mds = [
        VariableMetadata(1, 2, "finger_pos_x"),
        VariableMetadata(1, 2, "finger_pos_y"),
        VariableMetadata(1, 2, "box_pos_x"),
        VariableMetadata(1, 2, "box_pos_y"),
        VariableMetadata(1, 2, "ground_pos_x"),
        VariableMetadata(1, 2, "ground_pos_y"),
        VariableMetadata(1, 2, "finger_box_normal_force"),
        VariableMetadata(1, 2, "finger_box_friction_force"),
        VariableMetadata(1, 2, "box_ground_normal_force"),
        VariableMetadata(1, 2, "box_ground_friction_force"),
    ]

    # Create Bezier Curve
    curves = evaluate_curve_from_ctrl_points(vertex_values, var_mds)

    plt.plot(np.hstack(list(curves.values())))
    plt.legend(list(curves.keys()))
    animate_2d_box(**curves, box_width=box_width, box_height=box_height)

    return
