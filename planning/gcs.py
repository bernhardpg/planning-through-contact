import matplotlib.pyplot as plt
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
from pydrake.solvers import (
    MathematicalProgram,
    Solve,
    MathematicalProgramResult,
    L1NormCost,
    Cost,
    Binding,
)


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
        l1_norm_cost = L1NormCost(A, b)
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
        options.convex_relaxation = True  # TODO implement rounding
        options.max_rounded_paths = 10  # Must be >0 to actually do proper rounding

        result = self.gcs.SolveShortestPath(source, target, options)
        assert result.is_success
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
