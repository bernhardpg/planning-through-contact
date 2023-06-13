from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pydrake.geometry.optimization as opt
import pydrake.symbolic as sym
from pydrake.solvers import Binding, LinearCost

from geometry.collision_geometry.collision_geometry import (
    ContactLocation,
    PolytopeContactLocation,
)
from geometry.planar.planar_contact_modes import FaceContactMode, PlanarPlanSpecs
from geometry.planar.planar_pose import PlanarPose
from geometry.rigid_body import RigidBody
from tools.types import NpVariableArray


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

    def _formulate_contact_modes(self):
        contact_locations = self.slider.geometry.get_contact_locations()
        if not all([loc.pos == ContactLocation.FACE for loc in contact_locations]):
            raise RuntimeError("Only face contacts are supported for planar pushing.")

        self.contact_modes = [
            FaceContactMode.create_from_spec(loc, self.specs, self.slider)
            for loc in contact_locations
        ]

        # collision_free_regions = self.slider.geometry.get_collision_free_regions()
        # if not all([loc.pos == ContactLocation.FACE for loc in collision_free_regions]):
        #     raise RuntimeError("Only face contacts are supported for planar pushing.")

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
                breakpoint()
                vertex.AddCost(b)

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
