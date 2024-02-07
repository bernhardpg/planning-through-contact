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


# TODO(bernhardpg): This function should be somewhere else
def gcs_add_edge_with_continuity(
    gcs: opt.GraphOfConvexSets,
    outgoing: VertexModePair,
    incoming: VertexModePair,
    only_continuity_on_slider: bool = False,
    continuity_on_pusher_velocities: bool = False,
) -> GcsEdge:
    edge = gcs.AddEdge(outgoing.vertex, incoming.vertex)
    add_continuity_constraints_btwn_modes(
        outgoing.mode,
        incoming.mode,
        edge,
        only_continuity_on_slider,
        continuity_on_pusher_velocities,
    )
    return edge


@dataclass
class NonCollisionSubGraph:
    gcs: opt.GraphOfConvexSets
    sets: List[opt.ConvexSet]
    non_collision_modes: List[NonCollisionMode]
    non_collision_vertices: List[GcsVertex]
    slider: RigidBody
    config: PlanarPlanConfig
    source: Optional[VertexModePair] = None
    target: Optional[VertexModePair] = None

    @classmethod
    def create_with_gcs(
        cls,
        gcs: opt.GraphOfConvexSets,
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
        slider = config.dynamics_config.slider

        non_collision_modes = [
            NonCollisionMode.create_from_plan_spec(
                PolytopeContactLocation(ContactLocation.FACE, idx),
                config,
            )
            for idx in range(slider.geometry.num_collision_free_regions)
        ]

        vertex_names = [f"{subgraph_name}_{mode.name}" for mode in non_collision_modes]
        sets = [mode.get_convex_set() for mode in non_collision_modes]
        non_collision_vertices = [
            gcs.AddVertex(s, name) for s, name in zip(sets, vertex_names)
        ]

        for m, v in zip(non_collision_modes, non_collision_vertices):
            m.add_cost_to_vertex(v)

        edges = []

        # Add bi-directional edges
        edge_idxs = cls._get_overlapping_edge_idxs(non_collision_modes)
        for i, j in edge_idxs:
            if config.no_cycles:  # only connect lower idx faces to higher idx faces
                if i <= j:
                    e = gcs_add_edge_with_continuity(
                        gcs,
                        VertexModePair(
                            non_collision_vertices[i], non_collision_modes[i]
                        ),
                        VertexModePair(
                            non_collision_vertices[j], non_collision_modes[j]
                        ),
                        continuity_on_pusher_velocities=config.continuity_on_pusher_velocity,
                    )
                    edges.append(e)
                else:
                    e = gcs_add_edge_with_continuity(
                        gcs,
                        VertexModePair(
                            non_collision_vertices[j], non_collision_modes[j]
                        ),
                        VertexModePair(
                            non_collision_vertices[i], non_collision_modes[i]
                        ),
                        continuity_on_pusher_velocities=config.continuity_on_pusher_velocity,
                    )
                    edges.append(e)
            else:
                pair_u_1 = VertexModePair(
                    non_collision_vertices[i], non_collision_modes[i]
                )
                pair_v_1 = VertexModePair(
                    non_collision_vertices[j], non_collision_modes[j]
                )
                e_1 = gcs_add_edge_with_continuity(
                    gcs,
                    pair_u_1,
                    pair_v_1,
                    continuity_on_pusher_velocities=config.continuity_on_pusher_velocity,
                )

                pair_u_2 = VertexModePair(
                    non_collision_vertices[j], non_collision_modes[j]
                )
                pair_v_2 = VertexModePair(
                    non_collision_vertices[i], non_collision_modes[i]
                )
                e_2 = gcs_add_edge_with_continuity(
                    gcs,
                    VertexModePair(non_collision_vertices[j], non_collision_modes[j]),
                    VertexModePair(non_collision_vertices[i], non_collision_modes[i]),
                    continuity_on_pusher_velocities=config.continuity_on_pusher_velocity,
                )
                edges.append((pair_u_1, pair_v_1, e_1))
                edges.append((pair_u_2, pair_v_2, e_2))

        # Now we add all edge-specific costs (like the contact avoidance cost,
        # where we don't want to penalize the first/last knot point if the edge
        # is with a contact mode
        for pair_u, _, edge in edges:
            pair_u.mode.add_cost_to_edge(
                edge,
                pair_u.vertex,
                skip_first_knot_point=False,
                skip_last_knot_point=False,
            )

        return cls(
            gcs,
            sets,
            non_collision_modes,
            non_collision_vertices,
            slider,
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
            edge_incoming = gcs_add_edge_with_continuity(
                self.gcs, external_connection, subgraph_connection
            )

            assert isinstance(
                subgraph_connection.mode, NonCollisionMode
            )  # fix typing errors

            # For an edge (contact, noncontact) we do not penalize the first knot point
            # (which will be in contact)
            if isinstance(external_connection.mode, FaceContactMode):
                kwargs = {"skip_first_knot_point": True, "skip_last_knot_point": False}
            else:
                kwargs = {"skip_first_knot_point": False, "skip_last_knot_point": False}

            subgraph_connection.mode.add_cost_to_edge(
                edge_incoming,
                subgraph_connection.vertex,
                **kwargs,
            )
        if outgoing:
            edge_outgoing = gcs_add_edge_with_continuity(
                self.gcs, subgraph_connection, external_connection
            )

            assert isinstance(
                subgraph_connection.mode, NonCollisionMode
            )  # fix typing errors

            # For an edge (noncontact, contact) we do not penalize the last knot point
            # (which will be in contact)
            if isinstance(external_connection.mode, FaceContactMode):
                kwargs = {"skip_first_knot_point": False, "skip_last_knot_point": True}
            else:
                kwargs = {"skip_first_knot_point": False, "skip_last_knot_point": False}

            subgraph_connection.mode.add_cost_to_edge(
                edge_outgoing,
                subgraph_connection.vertex,
                **kwargs,
            )

    def _set_initial_or_final_poses(
        self,
        pusher_pose: PlanarPose,
        slider_pose: PlanarPose,
        initial_or_final: Literal["initial", "final"],
    ) -> None:
        mode = NonCollisionMode.create_source_or_target_mode(
            self.config, slider_pose, pusher_pose, initial_or_final
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
