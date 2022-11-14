from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pydot
import pydrake.geometry.optimization as opt
import pydrake.symbolic as sym
from pydrake.geometry.optimization import GraphOfConvexSets
from pydrake.math import eq
from pydrake.solvers import (
    Binding,
    Cost,
    L1NormCost,
    L2NormCost,
    MathematicalProgramResult,
)
from tqdm import tqdm

from geometry.bezier import BezierCurve
from geometry.collision_pair import CollisionPair, CollisionPairHandler
from geometry.rigid_body import RigidBody
from planning.graph_builder import ContactModeConfig, Graph, GraphBuilder


class GcsContactPlanner:
    def __init__(
        self,
        rigid_bodies: List[RigidBody],
        collision_pairs: List[CollisionPair],  # TODO Will be removed
        external_forces: List[sym.Expression],
        additional_constraints: Optional[List[sym.Formula]],
        allow_sliding: bool = False,
    ):
        self.rigid_bodies = rigid_bodies
        self.collision_pairs = collision_pairs

        self.all_decision_vars = self._collect_all_decision_vars(
            self.rigid_bodies, self.collision_pairs
        )
        self.all_position_vars = self.all_decision_vars[
            : self.num_bodies * (self.position_curve_order + 1) * self.position_dim
        ]
        self.all_force_vars = self.all_decision_vars[len(self.all_position_vars) :]

        self.collision_pair_handler = CollisionPairHandler(
            self.all_decision_vars,
            rigid_bodies,
            collision_pairs,
            external_forces,
            additional_constraints,
            allow_sliding,
        )

        self.graph_builder = GraphBuilder(self.collision_pair_handler)
        self.gcs = GraphOfConvexSets()

    def _collect_all_decision_vars(
        self,
        bodies: List[RigidBody],
        collision_pairs: List[CollisionPair],
    ) -> npt.NDArray[sym.Variable]:
        all_pos_vars = np.concatenate([b.pos.x.flatten() for b in bodies])
        all_normal_force_vars = np.concatenate(
            [p.lam_n.flatten() for p in collision_pairs]
        )
        all_friction_force_vars = np.concatenate(
            [p.lam_f.flatten() for p in collision_pairs]
        )
        all_vars = np.concatenate(
            [all_pos_vars, all_normal_force_vars, all_friction_force_vars]
        )
        return all_vars

    @property
    def num_bodies(self) -> int:
        return len(self.rigid_bodies)

    @property
    def num_pairs(self) -> int:
        return len(self.collision_pairs)

    @property
    def position_dim(self) -> int:
        return self.collision_pairs[0].body_a.dim

    @property
    def position_curve_order(self) -> int:
        return self.collision_pairs[0].body_a.position_curve_order

    @property
    def force_curve_order(self) -> int:
        return self.collision_pairs[0].force_curve_order

    @staticmethod
    def _find_source(
        vertex_map: Dict[str, GraphOfConvexSets.Vertex], source_name: str
    ) -> GraphOfConvexSets.Vertex:
        source = next(
            (vertex for name, vertex in vertex_map.items() if name == source_name),
            None,
        )
        if source is None:
            raise RuntimeError("No source node found!")
        return source

    @staticmethod
    def _find_target(
        vertex_map: Dict[str, GraphOfConvexSets.Vertex], target_name: str
    ) -> GraphOfConvexSets.Vertex:
        target = next(
            (vertex for name, vertex in vertex_map.items() if name == target_name),
            None,
        )
        if target is None:
            raise RuntimeError("No target node found!")
        return target

    def _formulate_graph(self, graph: Graph) -> None:
        print("Adding vertices...")
        vertex_map = {
            v.name: self.gcs.AddVertex(v.convex_set, v.name)
            for v in tqdm(graph.vertices)
        }

        # Retrieve source and target
        self.source = self._find_source(vertex_map, graph.source.name)
        self.target = self._find_source(vertex_map, graph.target.name)

        print("Adding edges...")
        for e in tqdm(graph.edges):
            u = vertex_map[e.u.name]
            v = vertex_map[e.v.name]
            self.gcs.AddEdge(u, v)

    def add_source_config(self, mc: ContactModeConfig) -> None:
        self.graph_builder.add_source_config(mc)

    def add_target_config(self, mc: ContactModeConfig) -> None:
        self.graph_builder.add_target_config(mc)

    def build_graph(self, prune: bool = False) -> None:
        graph = self.graph_builder.build_graph(prune)
        self._formulate_graph(graph)

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

    def _get_idxs_for_pos_ctrl_point_j(self, j: int) -> npt.NDArray[np.int32]:
        first_idxs = np.arange(0, self.position_dim * self.num_bodies) * (
            self.position_curve_order + 1
        )
        idxs = first_idxs + j
        return idxs

    def _get_idxs_for_force_ctrl_point_j(self, j: int) -> npt.NDArray[np.int32]:
        first_idxs = np.arange(0, 2 * self.num_pairs) * (self.force_curve_order + 1)
        idxs = first_idxs + j
        return idxs

    def add_position_continuity_constraints(self) -> None:
        first_idxs = self._get_idxs_for_pos_ctrl_point_j(0)
        last_idxs = self._get_idxs_for_pos_ctrl_point_j(self.position_curve_order)
        first_pos_vars = self.all_position_vars[first_idxs]
        last_pos_vars = self.all_position_vars[last_idxs]
        A_first = sym.DecomposeLinearExpressions(first_pos_vars, self.all_decision_vars)
        A_last = sym.DecomposeLinearExpressions(last_pos_vars, self.all_decision_vars)
        print("Adding position continuity constraints...")
        for e in tqdm(self.gcs.Edges()):
            xu, xv = e.xu(), e.xv()
            constraints = eq(A_last.dot(xu), A_first.dot(xv))
            for c in constraints:
                e.AddConstraint(c)

    def add_num_visited_vertices_cost(self, weight: float) -> None:
        print("Adding cost on number of visited vertices")
        for e in tqdm(self.gcs.Edges()):
            e.AddCost(weight)

    def add_force_strength_cost(self) -> None:
        A = sym.DecomposeLinearExpressions(self.all_force_vars, self.all_decision_vars)
        b = np.zeros((A.shape[0], 1))
        force_cost = L1NormCost(A, b)
        print("Adding force strength cost...")
        for e in tqdm(self.gcs.Edges()):
            cost = Binding[Cost](force_cost, e.xu())
            e.AddCost(cost)

    def add_position_path_length_cost(self) -> None:
        idxs = [
            self._get_idxs_for_pos_ctrl_point_j(j)
            for j in range(self.position_curve_order + 1)
        ]
        ctrl_point_diffs = np.diff(
            np.concatenate([[self.all_position_vars[i]] for i in idxs]), axis=0
        ).flatten()
        A = sym.DecomposeLinearExpressions(
            ctrl_point_diffs.flatten(), self.all_decision_vars
        )
        b = np.zeros((A.shape[0], 1))
        path_length_cost = L2NormCost(A, b)
        print("Adding position path length cost...")
        for v in tqdm(self.gcs.Vertices()):
            cost = Binding[Cost](path_length_cost, v.x())
            v.AddCost(cost)

    def add_force_path_length_cost(self) -> None:
        idxs = [
            self._get_idxs_for_force_ctrl_point_j(j)
            for j in range(self.position_curve_order + 1)
        ]
        ctrl_point_diffs = np.diff(
            np.concatenate([[self.all_force_vars[i]] for i in idxs]), axis=0
        ).flatten()
        A = sym.DecomposeLinearExpressions(
            ctrl_point_diffs.flatten(), self.all_decision_vars
        )
        b = np.zeros((A.shape[0], 1))
        force_length_cost = L2NormCost(A, b)
        print("Adding force path length cost...")
        for e in tqdm(self.gcs.Edges()):
            cost = Binding[Cost](force_length_cost, e.xu())
            e.AddCost(cost)

    def add_path_energy_cost(self) -> None:
        raise NotImplementedError
        ...
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

    def solve(self, use_convex_relaxation: bool = False) -> MathematicalProgramResult:
        options = opt.GraphOfConvexSetsOptions()
        options.convex_relaxation = use_convex_relaxation
        if use_convex_relaxation is True:
            options.preprocessing = True  # TODO Do I need to deal with this?
            options.max_rounded_paths = 10

        print("Solving GCS problem...")
        result = self.gcs.SolveShortestPath(self.source, self.target, options)
        assert result.is_success()
        print("Result is success!")
        return result

    def get_ctrl_points(
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
        self, vertex_values: npt.NDArray[np.float64], body: RigidBody
    ) -> List[npt.NDArray[np.float64]]:
        pos_ctrl_points = vertex_values[:, : len(self.all_position_vars)]
        num_pos_vars_per_body = self.position_dim * (self.position_curve_order + 1)
        body_idx = self.rigid_bodies.index(body) * num_pos_vars_per_body
        body_ctrl_points = pos_ctrl_points[
            :, body_idx : body_idx + num_pos_vars_per_body
        ]
        body_ctrl_points_reshaped = [
            c.reshape((self.position_dim, self.position_curve_order + 1))
            for c in body_ctrl_points
        ]
        return body_ctrl_points_reshaped

    def get_force_ctrl_points(self, vertex_values: List[npt.NDArray[np.float64]]):
        forces_ctrl_points = vertex_values[:, len(self.all_position_vars) :]
        # friction forces are always one dimensional
        num_force_vars_per_pair = self.force_curve_order + 1
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

    def get_curves_from_ctrl_points(
        self, vertex_values: npt.NDArray[np.float64]
    ) -> Tuple[
        Dict[str, npt.NDArray[np.float64]],
        Dict[str, npt.NDArray[np.float64]],
        Dict[str, npt.NDArray[np.float64]],
    ]:
        positions = {
            b.name: self.get_pos_ctrl_points(vertex_values, b)
            for b in self.rigid_bodies
        }
        pos_curves = {
            body: np.concatenate(
                [
                    BezierCurve.create_from_ctrl_points(c).eval_entire_interval()
                    for c in ctrl_points
                ]
            )
            for body, ctrl_points in positions.items()
        }

        normal_forces, friction_forces = self.get_force_ctrl_points(vertex_values)
        normal_force_curves = {
            pair: np.concatenate(
                [
                    BezierCurve.create_from_ctrl_points(
                        points.reshape((1, -1))
                    ).eval_entire_interval()
                    for points in control_points
                ]
            )
            for pair, control_points in normal_forces.items()
        }
        friction_force_curves = {
            pair: np.concatenate(
                [
                    BezierCurve.create_from_ctrl_points(
                        points.reshape((1, -1))
                    ).eval_entire_interval()
                    for points in control_points
                ]
            )
            for pair, control_points in friction_forces.items()
        }

        return pos_curves, normal_force_curves, friction_force_curves
