import itertools
from dataclasses import dataclass
from itertools import combinations
from typing import List, Literal, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pydot
import pydrake.geometry.optimization as opt
import pydrake.symbolic as sym
from pydrake.geometry.optimization import ConvexSet, GraphOfConvexSets
from pydrake.math import eq
from pydrake.solvers import (
    Binding,
    Cost,
    L1NormCost,
    L2NormCost,
    MathematicalProgramResult,
)

from geometry.contact import CollisionPair, ContactMode
from geometry.polyhedron import PolyhedronFormulator


@dataclass
class GcsContactPlanner:
    collision_pairs: List[CollisionPair]
    additional_constraints: List[sym.Formula]
    external_forces: npt.NDArray[sym.Expression]
    unactuated_bodies: List[str]

    @property
    def dim(self) -> int:
        return self.collision_pairs[0].body_a.dim

    @property
    def pos_order(self) -> int:
        return self.collision_pairs[0].body_a.order

    @property
    def num_bodies(self) -> int:
        return len(self.all_bodies)

    def __post_init__(self):
        self.gcs = GraphOfConvexSets()

        self.all_bodies = self._collect_all_rigid_bodies(self.collision_pairs)
        self.unactuated_dofs = self._get_unactuated_dofs(
            self.unactuated_bodies, self.all_bodies, self.dim
        )
        force_balance_constraints = self._construct_force_balance(
            self.collision_pairs,
            self.all_bodies,
            self.external_forces,
            self.unactuated_dofs,
        )

        for p in self.collision_pairs:
            p.add_force_balance(force_balance_constraints)

        for p in self.collision_pairs:
            p.add_constraint_to_all_modes(self.additional_constraints)

        self.all_decision_vars = self._collect_all_decision_vars(self.collision_pairs)
        # TODO this is now done twice
        self.all_pos_vars = self._collect_all_pos_vars(self.collision_pairs)

        for p in self.collision_pairs:
            p.formulate_contact_modes(self.all_decision_vars)

        convex_sets = self._create_all_convex_sets(self.collision_pairs)
        self._formulate_graph(convex_sets)

    # TODO make all functions static to adhere to pure functions?
    def _collect_all_rigid_bodies(self, pairs: List[CollisionPair]) -> List[str]:
        all_body_names = sorted(
            list(set(sum([[p.body_a.name, p.body_b.name] for p in pairs], [])))
        )
        return all_body_names

    def _collect_all_pos_vars(
        self, pairs: List[CollisionPair]
    ) -> npt.NDArray[sym.Variable]:
        all_pos_vars = np.array(
            sorted(
                list(
                    set(
                        np.concatenate(
                            [
                                np.concatenate(
                                    (p.body_a.pos.x, p.body_b.pos.x)
                                ).flatten()
                                for p in pairs
                            ]
                        )
                    )
                ),
                key=lambda x: x.get_name(),
            )
        )
        return all_pos_vars

    def _collect_all_decision_vars(
        self, pairs: List[CollisionPair]
    ) -> npt.NDArray[sym.Variable]:
        all_pos_vars = self._collect_all_pos_vars(pairs)
        all_normal_force_vars = np.concatenate([p.lam_n for p in pairs]).flatten()
        all_friction_force_vars = np.concatenate([p.lam_f for p in pairs]).flatten()
        all_vars = np.concatenate(
            [all_pos_vars, all_normal_force_vars, all_friction_force_vars]
        )
        return all_vars

    def _get_unactuated_dofs(
        self, unactuated_bodies: List[str], all_bodies: List[str], dim: int
    ) -> npt.NDArray[np.int32]:
        unactuated_idxs = [self.all_bodies.index(b) * dim for b in unactuated_bodies]
        unactuated_dofs = np.concatenate(
            [np.arange(idx, idx + dim) for idx in unactuated_idxs]
        )
        return unactuated_dofs

    def _construct_force_balance(
        self,
        collision_pairs: List[CollisionPair],
        bodies: List[str],
        external_forces: npt.NDArray[sym.Expression],
        unactuated_dofs: npt.NDArray[np.int32],
    ) -> List[sym.Formula]:
        normal_jacobians = np.vstack(
            [p.get_normal_jacobian_for_bodies(bodies) for p in collision_pairs]
        )
        tangential_jacobians = np.vstack(
            [p.get_tangential_jacobian_for_bodies(bodies) for p in collision_pairs]
        )

        normal_forces = np.concatenate([p.lam_n for p in collision_pairs])
        friction_forces = np.concatenate([p.lam_f for p in collision_pairs])

        all_force_balances = eq(
            normal_jacobians.T.dot(normal_forces)
            + tangential_jacobians.T.dot(friction_forces)
            + external_forces,
            0,
        )
        force_balance = all_force_balances[unactuated_dofs, :]
        return force_balance

    def _create_all_convex_sets(self, pairs: List[CollisionPair]) -> List[ConvexSet]:
        assert len(pairs) == 2  # TODO for now this only works for two pairs
        contact_pairs = [p.contact_modes for p in pairs]
        possible_contact_permutations = itertools.product(*contact_pairs)

        convex_sets = {
            f"{mode_1.name}_W_{mode_2.name}": mode_1.polyhedron.Intersection(
                mode_2.polyhedron
            )
            for (mode_1, mode_2) in possible_contact_permutations
            if mode_1.polyhedron.IntersectsWith(mode_2.polyhedron)
        }

        return convex_sets

    def _formulate_graph(self, convex_sets) -> None:
        for name, poly in convex_sets.items():
            self.gcs.AddVertex(poly, name)

        for u, v in itertools.permutations(self.gcs.Vertices(), 2):
            if u.set().IntersectsWith(v.set()):
                self.gcs.AddEdge(u, v)

    def allow_revisits_to_vertices(self, num_allowed_revisits: int) -> None:
        if num_allowed_revisits > 0:
            # TODO: This should be sped up, as it will scale poorly
            # Runtime: O(v * E), E ~= v^2, O(V^3)
            new_edges = []
            for _ in range(num_allowed_revisits):
                for v in self.gcs.Vertices():
                    # TODO very hardcoded, fix
                    if v.name() in ["source", "target"]:
                        continue
                    v_new = self.gcs.AddVertex(v.set(), f"{v.name()}_2")
                    for e in self.gcs.Edges():
                        if v == e.v():
                            new_edges.append((e.u(), v_new))
                        elif v == e.u():
                            new_edges.append((v_new, e.v()))

            for u, v in new_edges:
                self.gcs.AddEdge(u, v)

    def _find_path_to_target(
        self,
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
            return [u] + self._find_path_to_target(edges, target, v)

    def _create_intersecting_set_with_constraints(
        self,
        constraints: List[sym.Formula],
        all_vars: List[sym.Variable],
        vertices: GraphOfConvexSets.Vertices,
    ) -> Tuple[ConvexSet, GraphOfConvexSets.Vertex]:
        constraints_as_poly = PolyhedronFormulator(constraints).formulate_polyhedron(
            variables=all_vars, make_bounded=True
        )

        vertex = next(
            v for v in vertices if v.set().IntersectsWith(constraints_as_poly)
        )
        intersecting_set = constraints_as_poly.Intersection(vertex.set())
        return intersecting_set, vertex

    # TODO these are almost similar, clean up!
    def add_source(self, constraints: List[sym.Formula]):
        new_set, matching_vertex = self._create_intersecting_set_with_constraints(
            constraints, self.all_decision_vars, self.gcs.Vertices()
        )
        self.source = self.gcs.AddVertex(new_set, "source")
        self.gcs.AddEdge(self.source, matching_vertex)

    def add_target(self, constraints: List[sym.Formula]):
        new_set, matching_vertex = self._create_intersecting_set_with_constraints(
            constraints, self.all_decision_vars, self.gcs.Vertices()
        )
        self.target = self.gcs.AddVertex(new_set, "target")
        self.gcs.AddEdge(matching_vertex, self.target)

    def _get_idxs_for_pos_ctrl_point_j(
        self, flattened_pos_variables: npt.NDArray[sym.Variable], j: int
    ) -> npt.NDArray[np.int32]:
        first_idxs = np.arange(0, self.dim * self.num_bodies) * (self.pos_order + 1)
        idxs = first_idxs + j
        return idxs

    def add_position_continuity_constraints(self) -> None:
        first_idxs = self._get_idxs_for_pos_ctrl_point_j(self.all_pos_vars, 0)
        last_idxs = self._get_idxs_for_pos_ctrl_point_j(
            self.all_pos_vars, self.pos_order
        )
        first_pos_vars = self.all_pos_vars[first_idxs]
        last_pos_vars = self.all_pos_vars[last_idxs]
        A_first = sym.DecomposeLinearExpressions(first_pos_vars, self.all_decision_vars)
        A_last = sym.DecomposeLinearExpressions(last_pos_vars, self.all_decision_vars)
        for e in self.gcs.Edges():
            xu, xv = e.xu(), e.xv()
            constraints = eq(A_last.dot(xu), A_first.dot(xv))
            for c in constraints:
                e.AddConstraint(c)

    def add_num_visited_vertices_cost(self, weight: float) -> None:
        for e in self.gcs.Edges():
            e.AddCost(weight)

    def add_position_path_length_cost(self) -> None:
        idxs = [
            self._get_idxs_for_pos_ctrl_point_j(self.all_pos_vars, j)
            for j in range(self.pos_order + 1)
        ]
        ctrl_point_diffs = np.diff(
            np.concatenate([[self.all_pos_vars[i]] for i in idxs]), axis=0
        ).flatten()
        A = sym.DecomposeLinearExpressions(
            ctrl_point_diffs.flatten(), self.all_decision_vars
        )
        b = np.zeros((A.shape[0], 1))
        path_length_cost = L2NormCost(A, b)
        for v in self.gcs.Vertices():
            cost = Binding[Cost](path_length_cost, v.x())
            v.AddCost(cost)

        # Create path energy cost
        #    ADD_PATH_ENERGY_COST = False
        #    if ADD_PATH_ENERGY_COST:
        #        # PerspectiveQuadraticCost scales the cost by the
        #        # first element of z = Ax + b
        #        A_mod = np.vstack((np.zeros((1, A.shape[1])), A))
        #        b_mod = np.vstack((1, b))
        #        energy_cost = PerspectiveQuadraticCost(A_mod, b_mod)
        #        for e in gcs.Vertices():
        #            e_cost = Binding[Cost](energy_cost, e.xu())
        #            e.AddCost(e_cost)
        ...

    def solve(self, use_convex_relaxation: bool = False) -> MathematicalProgramResult:
        options = opt.GraphOfConvexSetsOptions()
        options.convex_relaxation = use_convex_relaxation
        if use_convex_relaxation is True:
            options.preprocessing = True  # TODO Do I need to deal with this?
            options.max_rounded_paths = 10

        result = self.gcs.SolveShortestPath(self.source, self.target, options)
        assert result.is_success()
        print("Result is success!")
        return result

    def get_vertex_values(
        self, result: MathematicalProgramResult
    ) -> List[npt.NDArray[np.float64]]:
        flow_variables = [e.phi() for e in self.gcs.Edges()]
        flow_results = [result.GetSolution(p) for p in flow_variables]
        active_edges = [
            edge for edge, flow in zip(self.gcs.Edges(), flow_results) if flow >= 0.99
        ]
        path = self._find_path_to_target(active_edges, self.target, self.source)
        vertex_values = [result.GetSolution(v.x()) for v in path]
        print("Path:")
        print([v.name() for v in path])
        return vertex_values

    def save_graph_diagram(
        self, filename: str, result: Optional[MathematicalProgramResult] = None
    ) -> None:
        if result is not None:
            graphviz = self.gcs.GetGraphvizString(result, False, precision=1)
        else:
            graphviz = self.gcs.GetGraphvizString()
        data = pydot.graph_from_dot_data(graphviz)[0]
        data.write_svg(filename)


# TODO should not be a dataclass
@dataclass
class GcsContactPlannerLEGACY:
    contact_mode_permutations: List[Tuple[ContactMode, ContactMode]]
    all_variables: npt.NDArray[sym.Variable]
    position_variables: npt.NDArray[sym.Variable]
    gcs: opt.GraphOfConvexSets = GraphOfConvexSets()

    def __post_init__(self):
        # TODO remove pos_sets
        self.pos_sets = [
            m1.pos_polyhedron.Intersection(m2.pos_polyhedron)
            for m1, m2 in self.contact_mode_permutations
        ]
        self.convex_sets = [
            m1.polyhedron.Intersection(m2.polyhedron)
            for m1, m2 in self.contact_mode_permutations
        ]
        sets_bounded = [s.IsBounded() for s in self.convex_sets]
        assert all(sets_bounded)

        self.names = [
            f"{m1.name}_AND_{m2.name}" for m1, m2 in self.contact_mode_permutations
        ]

        for name, convex_set in zip(self.names, self.convex_sets):
            self.gcs.AddVertex(convex_set, name)

        for (u, u_pos, u_modes), (v, v_pos, v_modes) in combinations(
            zip(self.gcs.Vertices(), self.pos_sets, self.contact_mode_permutations), 2
        ):
            if u_pos.IntersectsWith(v_pos):
                edge = self.gcs.AddEdge(u.id(), v.id())
                cont_constraints = self._create_position_continuity_constraints(edge)
                for c in cont_constraints:
                    edge.AddConstraint(c)

                l1_cost = self._create_position_path_length_cost(edge)
                edge.AddCost(l1_cost)

        self.save_graph_diagram("diagrams/graph.svg")

        breakpoint()
        # TODO remove

    #        for edge_def in self.edge_definitions:
    #            self._add_position_path_length_cost(edge_def["edge"], edge_def["mode_u"])

    def _create_position_continuity_constraints(
        self, edge: GraphOfConvexSets.Edge
    ) -> List[sym.Formula]:
        # TODO this can easily be sped up with bindings
        first_pos_vars = self.position_variables[:, 0]
        A_first = sym.DecomposeLinearExpressions(first_pos_vars, self.all_variables)
        last_pos_vars = self.position_variables[:, -1]
        A_last = sym.DecomposeLinearExpressions(last_pos_vars, self.all_variables)
        constraint = eq(A_last.dot(edge.xu()), A_first.dot(edge.xv()))
        return constraint

    def _create_position_path_length_cost(self, edge: GraphOfConvexSets.Edge) -> None:
        # Minimize euclidean distance between subsequent control points
        # NOTE: we only minimize L1 distance right now
        # TODO
        #        if mode_u.name == "source":  # no cost for source vertex
        #            return

        differences = (
            self.position_variables[:, 1:] - self.position_variables[:, :-1]
        ).flatten()
        A = sym.DecomposeLinearExpressions(differences, self.all_variables)
        b = np.zeros((A.shape[0], 1))
        l1_norm_cost = L1NormCost(A, b)  # TODO: This is just to have some cost
        c = Binding[Cost](l1_norm_cost, edge.xu())
        return c

    def _solve(
        self,
        source: GraphOfConvexSets.Vertex,
        target: GraphOfConvexSets.Vertex,
        convex_relaxation: bool = False,
    ) -> MathematicalProgramResult:
        options = opt.GraphOfConvexSetsOptions()
        options.convex_relaxation = convex_relaxation
        if convex_relaxation:
            options.preprocessing = True
            options.max_rounded_paths = 10  # Must be > 0 to actually do proper rounding

        result = self.gcs.SolveShortestPath(source, target, options)

        assert result.is_success()
        return result

    def _get_active_edges_in_result(
        self, edges: List[GraphOfConvexSets.Edge], result: MathematicalProgramResult
    ) -> List[GraphOfConvexSets.Edge]:
        flow_variables = [e.phi() for e in edges]
        flow_results = [result.GetSolution(p) for p in flow_variables]
        active_edges = [edge for edge, flow in zip(edges, flow_results) if flow == 1.00]
        return active_edges

    def _find_path_to_target(
        self,
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
            return [u] + self._find_path_to_target(edges, target, v)

    def _reconstruct_path(
        self, result: MathematicalProgramResult
    ) -> List[npt.NDArray[np.float64]]:
        active_edges = self._get_active_edges_in_result(self.gcs.Edges(), result)
        u = self.source
        path = self._find_path_to_target(active_edges, self.target, u)
        vertex_values = [result.GetSolution(v.x()) for v in path]
        return vertex_values

    def calculate_path(self) -> List[npt.NDArray[np.float64]]:
        assert self.source is not None
        assert self.target is not None
        self.solution = self._solve(self.source, self.target)
        vertex_values = self._reconstruct_path(self.solution)
        return vertex_values

    def set_source(self, source_name: str) -> None:
        source_vertex = next(v for v in self.gcs.Vertices() if v.name() == source_name)
        self.source = source_vertex

    def set_target(self, target_name: str) -> None:
        target_vertex = next(v for v in self.gcs.Vertices() if v.name() == target_name)
        self.target = target_vertex

    def save_graph_diagram(
        self,
        filename: str,
        use_solution: bool = False,
        show_binary_edge_vars: bool = False,
    ) -> None:
        if use_solution is False:
            graphviz = self.gcs.GetGraphvizString()
        else:
            assert self.solution is not None
            graphviz = self.gcs.GetGraphvizString(
                self.solution, show_binary_edge_vars, precision=1
            )

        data = pydot.graph_from_dot_data(graphviz)[0]
        data.write_svg(filename)


@dataclass
class GcsPlanner:
    order: int
    convex_sets: List[opt.ConvexSet]
    gcs: opt.GraphOfConvexSets = GraphOfConvexSets()

    def __post_init__(self):
        self.n_ctrl_points = self.order + 1
        self.dim = self.convex_sets[0].A().shape[1]

        for i, s in enumerate(self.convex_sets):
            self._create_vertex_from_set(s, f"v_{i}")

        self._create_edge_between_overlapping_sets()
        for e in self.gcs.Edges():
            print(f"Edge name: {e.name()}")

        for e in self.gcs.Edges():
            self._add_continuity_constraint(e)
            self._add_path_length_cost(e)

    def _create_vertex_from_set(self, s: opt.ConvexSet, name: str) -> None:
        # We need (order + 1) control variables within each set,
        # solve this by taking the Cartesian product of the set
        # with itself (order + 1) times:
        # A (x) A = [A 0;
        #            0 A]
        one_set_per_decision_var = s.CartesianPower(self.n_ctrl_points)
        # This creates ((order + 1) * dim) decision variables per vertex
        # which for 2D with order=2 will be ordered (x1, y1, x2, y2, x3, y3)
        self.gcs.AddVertex(one_set_per_decision_var, name)

    def _create_edge_between_overlapping_sets(self) -> None:
        for u, set_u in zip(self.gcs.Vertices(), self.convex_sets):
            for v, set_v in zip(self.gcs.Vertices(), self.convex_sets):
                # TODO this can be speed up as we dont need to check for overlap both ways
                if u == v:
                    continue
                sets_are_overlapping = set_u.IntersectsWith(set_v)
                if sets_are_overlapping:
                    self.gcs.AddEdge(u.id(), v.id(), f"({u.name()}, {v.name()})")

    def _add_continuity_constraint(self, edge: GraphOfConvexSets.Edge) -> None:
        u = edge.xu()  # (order + 1, dim)
        v = edge.xv()
        u_last_ctrl_point = u[-self.dim :]
        v_first_ctrl_point = v[: self.dim]
        # TODO: if we also want continuity to a higher degree, this needs to be enforced here!
        continuity_constraints = eq(u_last_ctrl_point, v_first_ctrl_point)
        for c in continuity_constraints:
            edge.AddConstraint(c)

    def _reshape_ctrl_points_to_matrix(
        self, vec: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        matr = vec.reshape((self.dim, self.n_ctrl_points), order="F")
        return matr

    def _add_path_length_cost(self, edge: GraphOfConvexSets.Edge) -> None:
        # Minimize euclidean distance between subsequent control points
        # NOTE: we only minimize squared distance right now (which is the same)
        ctrl_points = self._reshape_ctrl_points_to_matrix(edge.xu())
        differences = np.array(
            [ctrl_points[:, i + 1] - ctrl_points[:, i] for i in range(len(ctrl_points))]
        )
        A = sym.DecomposeLinearExpressions(differences.flatten(), edge.xu())
        b = np.zeros((A.shape[0], 1))
        l1_norm_cost = L1NormCost(A, b)  # TODO: This is just to have some cost
        edge.AddCost(Binding[Cost](l1_norm_cost, edge.xu()))
        # TODO: no cost for source vertex

    def add_point_vertex(
        self,
        p: npt.NDArray[np.float64],
        name: str,
        flow_direction: Literal["in", "out"],
    ) -> GraphOfConvexSets.Vertex:
        singleton_set = opt.Point(p)
        point_vertex = self.gcs.AddVertex(singleton_set, name)

        for v, set_v in zip(self.gcs.Vertices(), self.convex_sets):
            sets_are_overlapping = singleton_set.IntersectsWith(set_v)
            if sets_are_overlapping:
                if flow_direction == "out":
                    edge = self.gcs.AddEdge(
                        point_vertex.id(),
                        v.id(),
                        f"({point_vertex.name()}, {v.name()})",
                    )
                elif flow_direction == "in":
                    edge = self.gcs.AddEdge(
                        v.id(),
                        point_vertex.id(),
                        f"({v.name()}, {point_vertex.name()})",
                    )
                self._add_continuity_constraint(edge)
        return point_vertex

    def _solve(
        self, source: GraphOfConvexSets.Vertex, target: GraphOfConvexSets.Vertex
    ) -> MathematicalProgramResult:
        options = opt.GraphOfConvexSetsOptions()
        options.convex_relaxation = True
        options.max_rounded_paths = 10  # Must be >0 to actually do proper rounding

        result = self.gcs.SolveShortestPath(source, target, options)
        assert result.is_success()
        return result

    def _reconstruct_path(
        self, result: MathematicalProgramResult
    ) -> List[npt.NDArray[np.float64]]:
        edges = self.gcs.Edges()
        flow_variables = [e.phi() for e in edges]
        flow_results = [result.GetSolution(p) for p in flow_variables]
        active_edges = [edge for edge, flow in zip(edges, flow_results) if flow == 1.00]
        # Observe that we only need the first vertex in every edge to reconstruct the entire graph
        vertices_in_path = [edge.xu() for edge in active_edges]
        vertex_values = [result.GetSolution(v) for v in vertices_in_path]

        return vertex_values

    def calculate_path(
        self, source: GraphOfConvexSets.Vertex, target: GraphOfConvexSets.Vertex
    ) -> List[npt.NDArray[np.float64]]:
        result = self._solve(source, target)
        vertex_values = self._reconstruct_path(result)
        return vertex_values
