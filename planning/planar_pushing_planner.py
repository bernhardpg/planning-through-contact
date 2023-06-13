from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
import pydrake.geometry.optimization as opt
import pydrake.symbolic as sym
from pydrake.solvers import Binding, LinearCost

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


@dataclass
class NonCollisionSubGraph:
    vertices: List[GcsVertex]
    modes: List[NonCollisionMode]

    def get_edges(self) -> List[Tuple[GcsVertex, GcsVertex]]:
        """
        Returns all edges between any overlapping regions in the two-dimensional
        space of positions as a tuple of vertices
        """
        position_sets = [mode.get_convex_set_in_positions() for mode in self.modes]
        edge_idxs = [
            (i, j)
            for (i, u), (j, v) in combinations(enumerate(position_sets), 2)
            if u.IntersectsWith(v)
        ]
        edges = [(self.vertices[i], self.vertices[j]) for (i, j) in edge_idxs]
        return edges


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

        for mode_1, mode_2 in combinations(self.contact_modes, 2):
            self._build_subgraph(mode_1, mode_2)

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

        # TODO:
        # for each pair of contact modes:
        #   create a subgraph of noncollisionmodes connecting the contact modes

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

    def _build_subgraph(self, mode_1: FaceContactMode, mode_2: FaceContactMode):
        vertices = [mode.get_convex_set() for mode in self.non_collision_modes]
        subgraph = NonCollisionSubGraph(vertices, self.non_collision_modes)
        edges = subgraph.get_edges()
        breakpoint()

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
        self.source = self.gcs.AddVertex(point, name="source")
        # TODO: Cartesian product between slider and pusher initial pose

    def set_slider_target_pose(self, pose: PlanarPose) -> None:
        point = opt.Point(pose.vector())
        self.target = self.gcs.AddVertex(point, name="target")
        # TODO: Cartesian product between slider and pusher target pose

    def save_graph_diagram(self, filepath: Path) -> None:
        graphviz = self.gcs.GetGraphvizString()
        import pydot

        data = pydot.graph_from_dot_data(graphviz)[0]  # type: ignore
        data.write_svg(str(filepath))
