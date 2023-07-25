from typing import List, Tuple

import pydrake.geometry.optimization as opt
from pydrake.solvers import MathematicalProgramResult

GcsVertex = opt.GraphOfConvexSets.Vertex
GcsEdge = opt.GraphOfConvexSets.Edge
BidirGcsEdge = Tuple[GcsEdge, GcsEdge]


def _gcs_find_path_to_target(
    edges: List[GcsEdge], target: GcsVertex, u: GcsVertex
) -> List[GcsVertex]:
    current_edge = next(e for e in edges if e.u() == u)
    v = current_edge.v()
    target_reached = v == target
    if target_reached:
        return [u] + [v]
    else:
        return [u] + _gcs_find_path_to_target(edges, target, v)


def get_gcs_solution_path(
    gcs: opt.GraphOfConvexSets,
    result: MathematicalProgramResult,
    source_vertex: GcsVertex,
    target_vertex: GcsVertex,
    flow_treshold: float = 0.55,
) -> List[GcsVertex]:
    flow_variables = [e.phi() for e in gcs.Edges()]
    flow_results = [result.GetSolution(p) for p in flow_variables]
    active_edges = [
        edge for edge, flow in zip(gcs.Edges(), flow_results) if flow >= flow_treshold
    ]
    vertex_path = _gcs_find_path_to_target(active_edges, target_vertex, source_vertex)
    return vertex_path