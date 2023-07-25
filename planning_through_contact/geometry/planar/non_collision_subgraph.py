from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, NamedTuple, Tuple

import pydrake.geometry.optimization as opt
from pydrake.math import eq

from planning_through_contact.geometry.planar.face_contact import FaceContactMode
from planning_through_contact.geometry.planar.non_collision import NonCollisionMode

GcsVertex = opt.GraphOfConvexSets.Vertex
GcsEdge = opt.GraphOfConvexSets.Edge
BidirGcsEdge = Tuple[GcsEdge, GcsEdge]


class VertexModePair(NamedTuple):
    vertex: GcsVertex
    mode: FaceContactMode | NonCollisionMode


@dataclass
class NonCollisionSubGraph:
    sets: List[opt.ConvexSet]
    non_collision_modes: List[NonCollisionMode]
    vertices: List[GcsVertex]
    edges: Dict[Tuple[int, int], BidirGcsEdge]
    graph_connections: Dict[int, BidirGcsEdge]

    @classmethod
    def from_modes(
        cls,
        non_collision_modes: List[NonCollisionMode],
        gcs: opt.GraphOfConvexSets,
        first_contact_mode: int,
        second_contact_mode: int,
    ) -> "NonCollisionSubGraph":
        """
        Constructs a subgraph of non-collision modes, based on the given modes. This constructor takes in the GCS instance,
        as well as the modes. The modes each has the option to get a convex set from its underlying mathematical program,
        which is added as vertices to the GCS instance.

        An edge is added between any two overlapping position modes, as well as between the incoming and outgoing
        nodes to the bigger graph.

        @param first_contact_mode: Index of first contact mode where this subgraph is connected
        @param second_contact_mode: Index of second contact mode where this subgraph is connected
        """

        sets = [mode.get_convex_set() for mode in non_collision_modes]

        vertex_names = [
            f"{first_contact_mode}_TO_{second_contact_mode}_{mode.name}"
            for mode in non_collision_modes
        ]
        vertices = [gcs.AddVertex(s, name) for s, name in zip(sets, vertex_names)]

        edge_idxs = cls._get_overlapping_edge_idxs(non_collision_modes)
        # Add bi-directional edges
        edges = {
            (i, j): (
                gcs.AddEdge(vertices[i], vertices[j]),
                gcs.AddEdge(vertices[j], vertices[i]),
            )
            for i, j in edge_idxs
        }

        return cls(sets, non_collision_modes, vertices, edges, {})

    @staticmethod
    def _get_overlapping_edge_idxs(
        modes: List[NonCollisionMode],
    ) -> List[Tuple[int, int]]:
        """
        Returns all edges between any overlapping regions in the two-dimensional
        space of positions as a tuple of vertices
        """

        position_sets = [mode.get_convex_set_in_positions() for mode in modes]
        edge_idxs = [
            (i, j)
            for (i, u), (j, v) in combinations(enumerate(position_sets), 2)
            if u.IntersectsWith(v)
        ]
        return edge_idxs

    def __post_init__(self) -> None:
        for (i, j), (first_edge, second_edge) in self.edges.items():
            # Bi-directional edges
            self._add_continuity_constraints(i, j, first_edge)
            self._add_continuity_constraints(j, i, second_edge)

    def _add_continuity_constraints(
        self, outgoing_idx: int, incoming_idx: int, edge: GcsEdge
    ):
        first_vars = (
            self.non_collision_modes[incoming_idx].get_continuity_vars("first").vector
        )
        first_var_idxs = self.non_collision_modes[
            incoming_idx
        ].get_variable_indices_in_gcs_vertex(first_vars)

        last_vars = (
            self.non_collision_modes[outgoing_idx].get_continuity_vars("last").vector
        )
        last_var_idxs = self.non_collision_modes[
            outgoing_idx
        ].get_variable_indices_in_gcs_vertex(last_vars)

        constraint = eq(edge.xu()[last_var_idxs], edge.xv()[first_var_idxs])
        for c in constraint:
            edge.AddConstraint(c)

    def add_connection_to_full_graph(
        self,
        gcs: opt.GraphOfConvexSets,
        connection_vertex: GcsVertex,
        connection_idx: int,
    ) -> None:
        """
        Adds a bi-directional edge between the provided connection vertex and the subgraph vertex with index connection_idx in this subgraph.
        """

        self.graph_connections[connection_idx] = (
            gcs.AddEdge(connection_vertex, self.vertices[connection_idx]),
            gcs.AddEdge(self.vertices[connection_idx], connection_vertex),
        )

    def get_all_vertex_mode_pairs(self) -> Dict[str, VertexModePair]:
        return {
            v.name(): VertexModePair(vertex=v, mode=m)
            for v, m in zip(self.vertices, self.non_collision_modes)
        }
