from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
import pydrake.geometry.optimization as opt
import pydrake.symbolic as sym
from pydrake.solvers import Binding, LinearCost, MathematicalProgramResult

from geometry.collision_geometry.collision_geometry import (
    ContactLocation,
    PolytopeContactLocation,
)
from geometry.planar.planar_contact_modes import (
    FaceContactMode,
    NonCollisionMode,
    PlanarPlanSpecs,
)
from geometry.planar.planar_pose import PlanarPose
from geometry.rigid_body import RigidBody
from tools.types import NpVariableArray

GcsVertex = opt.GraphOfConvexSets.Vertex
GcsEdge = opt.GraphOfConvexSets.Edge


@dataclass
class NonCollisionSubGraph:
    sets: List[opt.ConvexSet]
    modes: List[NonCollisionMode]
    vertices: List[GcsVertex]
    edges: List[GcsEdge]
    graph_connections: List[GcsEdge]

    @classmethod
    def from_modes(
        cls,
        modes: List[NonCollisionMode],
        gcs: opt.GraphOfConvexSets,
        mode_i: int,
        mode_j: int,
    ) -> "NonCollisionSubGraph":
        """
        Constructs a subgraph of non-collision modes, based on the given modes. This constructor takes in the GCS instance,
        and adds the vertices and edges to the GCS instance.

        @param mode_i: Index of first contact mode where this subgraph is connected
        @param mode_j: Index of second contact mode where this subgraph is connected
        """

        sets = [mode.get_convex_set() for mode in modes]

        vertex_names = [f"{mode_i}_TO_{mode_j}_{mode.name}" for mode in modes]
        vertices = [gcs.AddVertex(s, name) for s, name in zip(sets, vertex_names)]

        edge_idxs = cls._get_edge_idxs(modes)
        # Add bi-directional edges
        edges = [gcs.AddEdge(vertices[i], vertices[j]) for i, j in edge_idxs]
        edges.extend([gcs.AddEdge(vertices[j], vertices[i]) for i, j in edge_idxs])

        no_graph_connections = []

        return cls(sets, modes, vertices, edges, no_graph_connections)

    @staticmethod
    def _get_edge_idxs(modes: List[NonCollisionMode]) -> List[Tuple[int, int]]:
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

    def add_connection_to_full_graph(
        self,
        gcs: opt.GraphOfConvexSets,
        connection_vertex: GcsVertex,
        connection_idx: int,
    ) -> None:
        """
        Adds a bi-directional edge between the provided connection vertex and the subgraph vertex with index connection_idx in this subgraph.
        """

        self.graph_connections.append(
            gcs.AddEdge(connection_vertex, self.vertices[connection_idx])
        )
        self.graph_connections.append(
            gcs.AddEdge(self.vertices[connection_idx], connection_vertex)
        )


@dataclass
class PlanarPushingPlanner:
    """
    A planner that generates motion plans for pushing an object (the "slider") with a point finger (the "pusher").
    The motion planner formulates the problem as a Graph-of-Convex-Sets problem, where each vertex in the graph
    corresponds to a contact mode.
    """

    slider: RigidBody
    specs: PlanarPlanSpecs

    def __post_init__(self):
        self.gcs = opt.GraphOfConvexSets()
        self._formulate_contact_modes()
        self._build_graph()
        self._add_costs()

    @property
    def num_contact_modes(self) -> int:
        return len(self.contact_modes)

    def _formulate_contact_modes(self):
        contact_locations = self.slider.geometry.contact_locations
        if not all([loc.pos == ContactLocation.FACE for loc in contact_locations]):
            raise RuntimeError("Only face contacts are supported for planar pushing.")

        self.contact_modes = [
            FaceContactMode.create_from_spec(loc, self.specs, self.slider)
            for loc in contact_locations
        ]

        self.non_collision_modes = [
            NonCollisionMode.create_from_spec(loc, self.specs, self.slider)
            for loc in contact_locations
        ]

    def _build_graph(self):
        self.contact_vertices = [
            self.gcs.AddVertex(mode.get_convex_set(), mode.name)
            for mode in self.contact_modes
        ]

        for mode_i, mode_j in combinations(range(self.num_contact_modes), 2):
            self._build_subgraph(mode_i, mode_j)

    def _add_costs(self):
        # Contact modes
        for mode, vertex in zip(self.contact_modes, self.contact_vertices):
            var_idxs, evaluators = mode.get_cost_terms()
            vars = vertex.x()[var_idxs]
            bindings = [Binding[LinearCost](e, v) for e, v in zip(evaluators, vars)]
            for b in bindings:
                vertex.AddCost(b)

        # Non collision modes
        # TODO:

    def _build_subgraph(self, mode_i: int, mode_j: int):
        subgraph = NonCollisionSubGraph.from_modes(
            self.non_collision_modes, self.gcs, mode_i, mode_j
        )
        subgraph.add_connection_to_full_graph(
            self.gcs, self.contact_vertices[mode_i], mode_i
        )
        subgraph.add_connection_to_full_graph(
            self.gcs, self.contact_vertices[mode_j], mode_j
        )

    def set_pusher_initial_pose(
        self, pose: PlanarPose, disregard_rotation: bool = True
    ) -> None:
        ...

    def set_pusher_final_pose(
        self, pose: PlanarPose, disregard_rotation: bool = True
    ) -> None:
        ...

    def set_slider_initial_pose(self, pose: PlanarPose) -> None:
        point = opt.Point(pose.vector())
        self.source_vertex = self.gcs.AddVertex(point, name="source")
        self.source_edges = [
            self.gcs.AddEdge(self.source_vertex, v) for v in self.contact_vertices
        ]
        # TODO: Cartesian product between slider and pusher initial pose

    def set_slider_target_pose(self, pose: PlanarPose) -> None:
        point = opt.Point(pose.vector())
        self.target_vertex = self.gcs.AddVertex(point, name="target")
        self.target_edges = [
            self.gcs.AddEdge(v, self.target_vertex) for v in self.contact_vertices
        ]
        # TODO: Cartesian product between slider and pusher target pose

    def solve(self) -> MathematicalProgramResult:
        result = self.gcs.SolveShortestPath(self.source_vertex, self.target_vertex)
        return result

    def save_graph_diagram(self, filepath: Path) -> None:
        graphviz = self.gcs.GetGraphvizString()
        import pydot

        data = pydot.graph_from_dot_data(graphviz)[0]  # type: ignore
        data.write_svg(str(filepath))
