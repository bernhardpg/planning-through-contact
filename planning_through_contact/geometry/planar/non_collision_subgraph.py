from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Literal, NamedTuple, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pydrake.geometry.optimization as opt
from pydrake.solvers import Binding, QuadraticCost

from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    ContactLocation,
    PolytopeContactLocation,
)
from planning_through_contact.geometry.planar.abstract_mode import (
    AbstractContactMode,
    add_continuity_constraints_btwn_modes,
)
from planning_through_contact.geometry.planar.face_contact import FaceContactMode
from planning_through_contact.geometry.planar.non_collision import NonCollisionMode
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.rigid_body import RigidBody
from planning_through_contact.planning.planar.planar_plan_config import PlanarPlanConfig

GcsVertex = opt.GraphOfConvexSets.Vertex
GcsEdge = opt.GraphOfConvexSets.Edge
BidirGcsEdge = Tuple[GcsEdge, GcsEdge]


class VertexModePair(NamedTuple):
    vertex: GcsVertex
    mode: AbstractContactMode


def gcs_add_edge_with_continuity(
    gcs: opt.GraphOfConvexSets,
    outgoing: VertexModePair,
    incoming: VertexModePair,
    only_continuity_on_slider: bool = False,
) -> None:
    edge = gcs.AddEdge(outgoing.vertex, incoming.vertex)
    add_continuity_constraints_btwn_modes(
        outgoing.mode, incoming.mode, edge, only_continuity_on_slider
    )


@dataclass
class NonCollisionSubGraph:
    gcs: opt.GraphOfConvexSets
    sets: List[opt.ConvexSet]
    non_collision_modes: List[NonCollisionMode]
    non_collision_vertices: List[GcsVertex]
    body: RigidBody
    config: PlanarPlanConfig
    source: Optional[VertexModePair] = None
    target: Optional[VertexModePair] = None

    @classmethod
    def create_with_gcs(
        cls,
        gcs: opt.GraphOfConvexSets,
        body: RigidBody,
        config: PlanarPlanConfig,
        subgraph_name: str,
    ) -> "NonCollisionSubGraph":
        """
        Constructs a subgraph of non-collision modes. An edge is added to
        the given gcs instance between any two overlapping position modes,
        with constraints that enforce continuity on poses.

        Squared euclidean distance is added as the cost on the finger position
        in all of the non-collision modes.
        """

        non_collision_modes = [
            NonCollisionMode.create_from_plan_spec(
                PolytopeContactLocation(ContactLocation.FACE, idx),
                config,
                body,
            )
            for idx in range(body.geometry.num_collision_free_regions)
        ]

        vertex_names = [f"{subgraph_name}_{mode.name}" for mode in non_collision_modes]
        sets = [mode.get_convex_set() for mode in non_collision_modes]
        non_collision_vertices = [
            gcs.AddVertex(s, name) for s, name in zip(sets, vertex_names)
        ]

        for m, v in zip(non_collision_modes, non_collision_vertices):
            m.add_cost_to_vertex(v)

        # Add bi-directional edges
        edge_idxs = cls._get_overlapping_edge_idxs(non_collision_modes)
        for i, j in edge_idxs:
            gcs_add_edge_with_continuity(
                gcs,
                VertexModePair(non_collision_vertices[i], non_collision_modes[i]),
                VertexModePair(non_collision_vertices[j], non_collision_modes[j]),
            )
            gcs_add_edge_with_continuity(
                gcs,
                VertexModePair(non_collision_vertices[j], non_collision_modes[j]),
                VertexModePair(non_collision_vertices[i], non_collision_modes[i]),
            )

        return cls(
            gcs,
            sets,
            non_collision_modes,
            non_collision_vertices,
            body,
            config,
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
        incoming: bool = True,
        outgoing: bool = True,
    ) -> None:
        subgraph_connection = VertexModePair(
            self.non_collision_vertices[subgraph_connection_idx],
            self.non_collision_modes[subgraph_connection_idx],
        )
        if incoming:
            gcs_add_edge_with_continuity(
                self.gcs, external_connection, subgraph_connection
            )
        if outgoing:
            gcs_add_edge_with_continuity(
                self.gcs, subgraph_connection, external_connection
            )

    def _set_initial_or_final_poses(
        self,
        pusher_pose: PlanarPose,
        slider_pose: PlanarPose,
        initial_or_final: Literal["initial", "final"],
    ) -> None:
        mode = NonCollisionMode.create_source_or_target_mode(
            self.config, slider_pose, pusher_pose, self.body, initial_or_final
        )
        vertex = self.gcs.AddVertex(mode.get_convex_set(), mode.name)

        pair = VertexModePair(vertex, mode)
        if initial_or_final == "initial":
            kwargs = {"outgoing": False, "incoming": True}
        else:
            kwargs = {"outgoing": True, "incoming": False}
        self.connect_with_continuity_constraints(
            mode.contact_location.idx, pair, **kwargs
        )

        if initial_or_final == "initial":
            self.source = pair
        else:  # final
            self.target = pair

    def set_initial_poses(
        self,
        pusher_initial_pose: PlanarPose,
        slider_initial_pose: PlanarPose,
    ) -> None:
        self._set_initial_or_final_poses(
            pusher_initial_pose, slider_initial_pose, "initial"
        )

    def set_final_poses(
        self,
        pusher_final_pose: PlanarPose,
        slider_final_pose: PlanarPose,
    ) -> None:
        self._set_initial_or_final_poses(pusher_final_pose, slider_final_pose, "final")

    def get_all_vertex_mode_pairs(self) -> Dict[str, VertexModePair]:
        all_pairs = {
            v.name(): VertexModePair(vertex=v, mode=m)
            for v, m in zip(self.non_collision_vertices, self.non_collision_modes)
        }
        if self.source:
            all_pairs[self.source.mode.name] = self.source
        if self.target:
            all_pairs[self.target.mode.name] = self.target

        return all_pairs
