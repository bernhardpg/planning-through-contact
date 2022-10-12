import itertools
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pydot
import pydrake.geometry.optimization as opt
import pydrake.symbolic as sym
from pydrake.geometry.optimization import ConvexSet, GraphOfConvexSets
from pydrake.math import eq
from pydrake.solvers import Binding, Cost, L2NormCost, MathematicalProgramResult

from geometry.contact import CollisionPair
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
    def force_order(self) -> int:
        # TODO ambiguous that here we use np array directly and above we use BezierVariable
        return self.collision_pairs[0].lam_n.shape[1] - 1

    @property
    def num_bodies(self) -> int:
        return len(self.all_bodies)

    @property
    def num_pairs(self) -> int:
        return len(self.collision_pairs)

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

    def _create_intersecting_set_with_first_matching_vertex(
        self,
        constraints: List[sym.Formula],
        all_vars: List[sym.Variable],
        vertices: GraphOfConvexSets.Vertices,
    ) -> Tuple[ConvexSet, GraphOfConvexSets.Vertex]:
        constraints_as_poly = PolyhedronFormulator(constraints).formulate_polyhedron(
            variables=all_vars, make_bounded=True
        )

        vertices_matching_constraints = [
            v for v in vertices if v.set().IntersectsWith(constraints_as_poly)
        ]
        if len(vertices_matching_constraints) == 0:
            raise ValueError("No vertices match given constraints.")
        vertex = vertices_matching_constraints[0]
        intersecting_set = constraints_as_poly.Intersection(vertex.set())
        return intersecting_set, vertex

    # TODO these are almost similar, clean up!
    def add_source(self, constraints: List[sym.Formula]):
        (
            new_set,
            matching_vertex,
        ) = self._create_intersecting_set_with_first_matching_vertex(
            constraints, self.all_decision_vars, self.gcs.Vertices()
        )
        self.source = self.gcs.AddVertex(new_set, "source")
        self.gcs.AddEdge(self.source, matching_vertex)

    def add_target(self, constraints: List[sym.Formula]):
        (
            new_set,
            matching_vertex,
        ) = self._create_intersecting_set_with_first_matching_vertex(
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

        # TODO
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
    ) -> npt.NDArray[np.float64]:
        flow_variables = [e.phi() for e in self.gcs.Edges()]
        flow_results = [result.GetSolution(p) for p in flow_variables]
        active_edges = [
            edge for edge, flow in zip(self.gcs.Edges(), flow_results) if flow >= 0.99
        ]
        path = self._find_path_to_target(active_edges, self.target, self.source)
        vertex_values = np.vstack([result.GetSolution(v.x()) for v in path])
        print("Path:")
        print([v.name() for v in path])
        return vertex_values

    def get_pos_ctrl_points(
        self, vertex_values: npt.NDArray[np.float64], body_name: str
    ) -> List[npt.NDArray[np.float64]]:
        pos_ctrl_points = vertex_values[:, : len(self.all_pos_vars)]
        num_pos_vars_per_body = self.dim * (self.pos_order + 1)
        body_idx = self.all_bodies.index(body_name) * num_pos_vars_per_body
        body_ctrl_points = pos_ctrl_points[
            :, body_idx : body_idx + num_pos_vars_per_body
        ]
        body_ctrl_points_reshaped = [
            c.reshape((self.dim, self.pos_order + 1)) for c in body_ctrl_points
        ]
        return body_ctrl_points_reshaped

    def get_force_ctrl_points(self, vertex_values: List[npt.NDArray[np.float64]]):
        forces_ctrl_points = vertex_values[:, len(self.all_pos_vars) :]
        # friction forces are always one dimensional
        num_force_vars_per_pair = self.force_order + 1
        normal_forces_ctrl_points, friction_forces_ctrl_points = np.split(
            forces_ctrl_points, [num_force_vars_per_pair * self.num_pairs], axis=1
        )
        normal_forces = {}
        friction_forces = {}
        for idx, p in enumerate(self.collision_pairs):
            n_force = normal_forces_ctrl_points[
                :, idx * num_force_vars_per_pair : (idx + 1) * num_force_vars_per_pair
            ]
            normal_forces[p.name] = n_force
            f_force = friction_forces_ctrl_points[
                :, idx * num_force_vars_per_pair : (idx + 1) * num_force_vars_per_pair
            ]
            friction_forces[p.name] = f_force

        return normal_forces, friction_forces

    def save_graph_diagram(
        self, filename: str, result: Optional[MathematicalProgramResult] = None
    ) -> None:
        if result is not None:
            graphviz = self.gcs.GetGraphvizString(result, False, precision=1)
        else:
            graphviz = self.gcs.GetGraphvizString()
        data = pydot.graph_from_dot_data(graphviz)[0]
        data.write_svg(filename)
