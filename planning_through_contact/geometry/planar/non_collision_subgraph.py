from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, NamedTuple, Tuple

import pydrake.geometry.optimization as opt
from pydrake.math import eq

from planning_through_contact.geometry.planar.abstract_mode import (
    add_continuity_constraints_btwn_modes,
)
from planning_through_contact.geometry.planar.face_contact import FaceContactMode
from planning_through_contact.geometry.planar.non_collision import NonCollisionMode
from planning_through_contact.geometry.rigid_body import RigidBody
from planning_through_contact.planning.planar.planar_plan_specs import PlanarPlanSpecs

GcsVertex = opt.GraphOfConvexSets.Vertex
GcsEdge = opt.GraphOfConvexSets.Edge
BidirGcsEdge = Tuple[GcsEdge, GcsEdge]


class VertexModePair(NamedTuple):
    vertex: GcsVertex
    mode: FaceContactMode | NonCollisionMode


def gcs_add_edge_with_continuity(
    gcs: opt.GraphOfConvexSets, outgoing: VertexModePair, incoming: VertexModePair
) -> None:
    edge = gcs.AddEdge(outgoing.vertex, incoming.vertex)
    add_continuity_constraints_btwn_modes(outgoing.mode, incoming.mode, edge)


@dataclass
class NonCollisionSubGraph:
    gcs: opt.GraphOfConvexSets
    sets: List[opt.ConvexSet]
    non_collision_modes: List[NonCollisionMode]
    vertices: List[GcsVertex]

    @classmethod
    def create_with_gcs(
        cls,
        gcs: opt.GraphOfConvexSets,
        body: RigidBody,
        plan_specs: PlanarPlanSpecs,
        subgraph_name: str,
    ) -> "NonCollisionSubGraph":
        """
        Constructs a subgraph of non-collision modes, based on the given modes. This constructor takes in the GCS instance,
        as well as the modes. The modes each has the option to get a convex set from its underlying mathematical program,
        which is added as vertices to the GCS instance.

        An edge is added between any two overlapping position modes, as well as between the incoming and outgoing
        nodes to the bigger graph.

        @param first_contact_mode: First contact mode where this subgraph is connected
        @param second_contact_mode: Second contact mode where this subgraph is connected
        """

        non_collision_modes = [
            NonCollisionMode.create_from_plan_spec(loc, plan_specs, body)
            for loc in body.geometry.contact_locations
        ]

        sets = [mode.get_convex_set() for mode in non_collision_modes]

        vertex_names = [f"{subgraph_name}_{mode.name}" for mode in non_collision_modes]
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
        for (i, j), (first_edge, second_edge) in edges.items():
            add_continuity_constraints_btwn_modes(
                non_collision_modes[i], non_collision_modes[j], first_edge
            )
            add_continuity_constraints_btwn_modes(
                non_collision_modes[j], non_collision_modes[j], second_edge
            )

        return cls(
            gcs,
            sets,
            non_collision_modes,
            vertices,
        )

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

    def connect_with_continuity_constraints(
        self,
        subgraph_connection_idx: int,
        external_connection: VertexModePair,
    ) -> None:
        subgraph_connection = VertexModePair(
            self.vertices[subgraph_connection_idx],
            self.non_collision_modes[subgraph_connection_idx],
        )
        # bi-directional edges
        gcs_add_edge_with_continuity(self.gcs, external_connection, subgraph_connection)
        gcs_add_edge_with_continuity(self.gcs, subgraph_connection, external_connection)

    def get_all_vertex_mode_pairs(self) -> Dict[str, VertexModePair]:
        return {
            v.name(): VertexModePair(vertex=v, mode=m)
            for v, m in zip(self.vertices, self.non_collision_modes)
        }