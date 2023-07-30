from itertools import combinations
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import pydrake.geometry.optimization as opt
from pydrake.solvers import CommonSolverOption, MathematicalProgramResult, SolverOptions

from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    ContactLocation,
    PolytopeContactLocation,
)
from planning_through_contact.geometry.planar.abstract_mode import AbstractModeVariables
from planning_through_contact.geometry.planar.face_contact import FaceContactMode
from planning_through_contact.geometry.planar.non_collision import NonCollisionMode
from planning_through_contact.geometry.planar.non_collision_subgraph import (
    NonCollisionSubGraph,
    VertexModePair,
    gcs_add_edge_with_continuity,
)
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.planar.planar_pushing_path import (
    PlanarPushingPath,
)
from planning_through_contact.geometry.planar.trajectory_builder import (
    PlanarPushingTrajectory,
    PlanarTrajectoryBuilder,
)
from planning_through_contact.geometry.rigid_body import RigidBody
from planning_through_contact.planning.planar.planar_plan_specs import PlanarPlanSpecs

GcsVertex = opt.GraphOfConvexSets.Vertex
GcsEdge = opt.GraphOfConvexSets.Edge
BidirGcsEdge = Tuple[GcsEdge, GcsEdge]


class PlanarPushingPlanner:
    """
    A planner that generates motion plans for pushing an object (the "slider") with a point finger (the "pusher").
    The motion planner formulates the problem as a Graph-of-Convex-Sets problem, where each vertex in the graph
    corresponds to a contact mode.
    """

    cost_param_transition_cost: float = 0.1

    def __init__(
        self,
        slider: RigidBody,
        plan_specs: PlanarPlanSpecs,
        contact_locations: Optional[List[PolytopeContactLocation]] = None,
        avoid_object: bool = False,
        allow_teleportation: bool = False,
        penalize_mode_transition: bool = False,
    ):
        self.slider = slider
        self.plan_specs = plan_specs
        self.avoid_object = avoid_object
        self.allow_teleportation = allow_teleportation
        self.penalize_mode_transition = penalize_mode_transition

        self.source = None
        self.target = None

        if self.avoid_object and plan_specs.num_knot_points_non_collision <= 2:
            raise ValueError(
                "It is not possible to avoid object with only 2 knot points."
            )

        if self.avoid_object and self.allow_teleportation:
            raise ValueError("Cannot avoid object while allowing teleportation")

        # TODO(bernhardpg): should just extract faces, rather than relying on the
        # object to only pass faces as contact locations
        self.contact_locations = contact_locations
        if self.contact_locations is None:
            self.contact_locations = slider.geometry.contact_locations

        self.gcs = opt.GraphOfConvexSets()
        self._formulate_contact_modes()
        self._build_graph()

        # costs for non-collisions are added by each of the separate subgraphs
        for m, v in zip(self.contact_modes, self.contact_vertices):
            m.add_cost_to_vertex(v)

        if self.penalize_mode_transition:
            for v in self.contact_vertices:
                v.AddCost(self.cost_param_transition_cost)

    @property
    def num_contact_modes(self) -> int:
        return len(self.contact_modes)

    def _formulate_contact_modes(self):
        assert self.contact_locations is not None

        if not all([loc.pos == ContactLocation.FACE for loc in self.contact_locations]):
            raise RuntimeError("Only face contacts are supported for planar pushing.")

        self.contact_modes = [
            FaceContactMode.create_from_plan_spec(loc, self.plan_specs, self.slider)
            for loc in self.contact_locations
        ]

    def _build_graph(self):
        self.contact_vertices = [
            self.gcs.AddVertex(mode.get_convex_set(), mode.name)
            for mode in self.contact_modes
        ]

        if self.allow_teleportation:
            for i, j in combinations(range(self.num_contact_modes), 2):
                gcs_add_edge_with_continuity(
                    self.gcs,
                    VertexModePair(self.contact_vertices[i], self.contact_modes[i]),
                    VertexModePair(self.contact_vertices[j], self.contact_modes[j]),
                    only_continuity_on_slider=True,
                )
                gcs_add_edge_with_continuity(
                    self.gcs,
                    VertexModePair(self.contact_vertices[j], self.contact_modes[j]),
                    VertexModePair(self.contact_vertices[i], self.contact_modes[i]),
                    only_continuity_on_slider=True,
                )
        else:
            # connect contact modes through NonCollisionSubGraphs
            self.subgraphs = [
                self._build_subgraph_between_contact_modes(mode_i, mode_j)
                for mode_i, mode_j in combinations(range(self.num_contact_modes), 2)
            ]

            self.source_subgraph = self._create_entry_or_exit_subgraph("entry")
            self.target_subgraph = self._create_entry_or_exit_subgraph("exit")

    def _build_subgraph_between_contact_modes(
        self, first_contact_mode_idx: int, second_contact_mode_idx: int
    ) -> NonCollisionSubGraph:
        subgraph = NonCollisionSubGraph.create_with_gcs(
            self.gcs,
            self.slider,
            self.plan_specs,
            f"FACE_{first_contact_mode_idx}_to_FACE_{second_contact_mode_idx}",
            avoid_object=self.avoid_object,
        )
        for idx in (first_contact_mode_idx, second_contact_mode_idx):
            subgraph.connect_with_continuity_constraints(
                idx,
                VertexModePair(
                    self.contact_vertices[idx],
                    self.contact_modes[idx],
                ),
            )
        return subgraph

    def _get_all_vertex_mode_pairs(self) -> Dict[str, VertexModePair]:
        all_pairs = {
            v.name(): VertexModePair(vertex=v, mode=m)
            for v, m in zip(self.contact_vertices, self.contact_modes)
        }
        if self.allow_teleportation:
            assert self.source is not None
            assert self.target is not None

            all_pairs[self.source.mode.name] = self.source
            all_pairs[self.target.mode.name] = self.target
        else:
            for subgraph in self.subgraphs:
                all_pairs.update(subgraph.get_all_vertex_mode_pairs())

            for subgraph in (self.source_subgraph, self.target_subgraph):
                all_pairs.update(subgraph.get_all_vertex_mode_pairs())

        return all_pairs

    def _create_entry_or_exit_subgraph(
        self, entry_or_exit: Literal["entry", "exit"]
    ) -> NonCollisionSubGraph:
        if entry_or_exit == "entry":
            name = "ENTRY"
            kwargs = {"outgoing": True, "incoming": False}
        else:
            name = "EXIT"
            kwargs = {"outgoing": False, "incoming": True}

        subgraph = NonCollisionSubGraph.create_with_gcs(
            self.gcs, self.slider, self.plan_specs, name, avoid_object=self.avoid_object
        )

        for idx, (vertex, mode) in enumerate(
            zip(self.contact_vertices, self.contact_modes)
        ):
            subgraph.connect_with_continuity_constraints(
                idx, VertexModePair(vertex, mode), **kwargs
            )
        return subgraph

    def set_initial_poses(
        self,
        finger_pose: PlanarPose,
        slider_pose: PlanarPose,
    ) -> None:
        self.finger_pose_initial = finger_pose
        self.slider_pose_initial = slider_pose

        if self.allow_teleportation:
            self.source = self._add_single_source_or_target(
                finger_pose, slider_pose, "initial"
            )
        else:
            self.source_subgraph.set_initial_poses(finger_pose, slider_pose)
            self.source = self.source_subgraph.source

    def set_target_poses(
        self,
        finger_pose: PlanarPose,
        slider_pose: PlanarPose,
    ) -> None:
        self.finger_pose_target = finger_pose
        self.slider_pose_target = slider_pose

        if self.allow_teleportation:
            self.target = self._add_single_source_or_target(
                finger_pose, slider_pose, "final"
            )
        else:
            self.target_subgraph.set_final_poses(finger_pose, slider_pose)
            self.target = self.target_subgraph.target

    def _add_single_source_or_target(
        self,
        finger_pose: PlanarPose,
        slider_pose: PlanarPose,
        initial_or_final: Literal["initial", "final"],
    ) -> VertexModePair:
        mode = NonCollisionMode.create_source_or_target_mode(
            self.plan_specs, slider_pose, finger_pose, self.slider, initial_or_final
        )
        vertex = self.gcs.AddVertex(mode.get_convex_set(), mode.name)
        pair = VertexModePair(vertex, mode)

        # connect source or target to all contact modes
        if initial_or_final == "initial":
            # source to contact modes
            for contact_vertex, contact_mode in zip(
                self.contact_vertices, self.contact_modes
            ):
                gcs_add_edge_with_continuity(
                    self.gcs,
                    pair,
                    VertexModePair(contact_vertex, contact_mode),
                    only_continuity_on_slider=True,
                )
        else:  # contact modes to target
            for contact_vertex, contact_mode in zip(
                self.contact_vertices, self.contact_modes
            ):
                gcs_add_edge_with_continuity(
                    self.gcs,
                    VertexModePair(contact_vertex, contact_mode),
                    pair,
                    only_continuity_on_slider=True,
                )

        return pair

    def _solve(
        self, print_output: bool = False, convex_relaxation: bool = True
    ) -> MathematicalProgramResult:
        options = opt.GraphOfConvexSetsOptions()
        if print_output:
            options.solver_options = SolverOptions()
            options.solver_options.SetOption(CommonSolverOption.kPrintToConsole, 1)  # type: ignore

        options.convex_relaxation = convex_relaxation
        if options.convex_relaxation is True:
            options.preprocessing = True  # TODO(bernhardpg): should this be changed?
            options.max_rounded_paths = 1

        assert self.source is not None
        assert self.target is not None

        result = self.gcs.SolveShortestPath(
            self.source.vertex, self.target.vertex, options
        )
        return result

    def get_vertex_solution_path(
        self,
        result: MathematicalProgramResult,
    ) -> List[GcsVertex]:
        """
        Returns the vertices on the solution path in the correct order,
        given a MathematicalProgramResult.
        """
        path = self.get_solution_path(result)
        return path.get_vertices()

    def get_solution_path(self, result: MathematicalProgramResult) -> PlanarPushingPath:
        assert self.source is not None
        assert self.target is not None

        path = PlanarPushingPath.from_result(
            self.gcs,
            result,
            self.source.vertex,
            self.target.vertex,
            self._get_all_vertex_mode_pairs(),
        )
        return path

    def plan_trajectory(
        self,
        print_output: bool = False,
        measure_time: bool = False,
        interpolate: bool = True,
        round_trajectory: bool = False,
    ) -> PlanarPushingTrajectory:
        assert self.source is not None
        assert self.target is not None

        import time

        start = time.time()
        result = self._solve(print_output)
        end = time.time()

        assert result.is_success()

        if measure_time:
            elapsed_time = end - start
            print(f"Total elapsed optimization time: {elapsed_time}")

        path = self.get_solution_path(result)
        vars_on_path = path.get_rounded_vars() if round_trajectory else path.get_vars()
        traj = PlanarTrajectoryBuilder(vars_on_path).get_trajectory(
            interpolate=interpolate
        )

        return traj

    def save_graph_diagram(self, filepath: Path) -> None:
        graphviz = self.gcs.GetGraphvizString()
        import pydot

        data = pydot.graph_from_dot_data(graphviz)[0]  # type: ignore
        data.write_svg(str(filepath))
