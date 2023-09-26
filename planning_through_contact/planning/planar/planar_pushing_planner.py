from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import pydot
import pydrake.geometry.optimization as opt
from pydrake.solvers import (
    CommonSolverOption,
    MathematicalProgramResult,
    MosekSolver,
    SolverOptions,
)

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
from planning_through_contact.geometry.planar.planar_pushing_trajectory import (
    PlanarPushingTrajectory,
)
from planning_through_contact.geometry.rigid_body import RigidBody
from planning_through_contact.geometry.utilities import two_d_rotation_matrix_from_angle
from planning_through_contact.planning.planar.planar_plan_config import (
    PlanarPlanConfig,
    PlanarSolverParams,
)

GcsVertex = opt.GraphOfConvexSets.Vertex
GcsEdge = opt.GraphOfConvexSets.Edge
BidirGcsEdge = Tuple[GcsEdge, GcsEdge]


@dataclass
class PlanarPushingStartAndGoal:
    slider_initial_pose: PlanarPose
    slider_target_pose: PlanarPose
    pusher_initial_pose: PlanarPose
    pusher_target_pose: PlanarPose

    def rotate(self, theta: float) -> "PlanarPushingStartAndGoal":
        new_slider_init = self.slider_initial_pose.rotate(theta)
        new_slider_target = self.slider_target_pose.rotate(theta)

        # NOTE: Pusher poses are already relative to slider frame, not world frame
        return PlanarPushingStartAndGoal(
            new_slider_init,
            new_slider_target,
            self.pusher_initial_pose,
            self.pusher_target_pose,
        )


class PlanarPushingPlanner:
    """
    A planner that generates motion plans for pushing an object (the "slider") with a point finger (the "pusher").
    The motion planner formulates the problem as a Graph-of-Convex-Sets problem, where each vertex in the graph
    corresponds to a contact mode.
    """

    cost_param_transition_cost: float = 0.5

    def __init__(
        self,
        config: PlanarPlanConfig,
        contact_locations: Optional[List[PolytopeContactLocation]] = None,
    ):
        self.slider = config.dynamics_config.slider
        self.config = config

        self.source = None
        self.target = None

        if self.config.avoid_object and config.num_knot_points_non_collision <= 2:
            raise ValueError(
                "It is not possible to avoid object with only 2 knot points."
            )

        if self.config.avoid_object and self.config.allow_teleportation:
            raise ValueError("Cannot avoid object while allowing teleportation")

        # TODO(bernhardpg): should just extract faces, rather than relying on the
        # object to only pass faces as contact locations
        self.contact_locations = contact_locations
        if self.contact_locations is None:
            self.contact_locations = self.slider.geometry.contact_locations

        self.gcs = opt.GraphOfConvexSets()
        self._formulate_contact_modes()
        self._build_graph()

        # costs for non-collisions are added by each of the separate subgraphs
        for m, v in zip(self.contact_modes, self.contact_vertices):
            m.add_cost_to_vertex(v)

        if self.config.penalize_mode_transitions:
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
            FaceContactMode.create_from_plan_spec(
                loc,
                self.config,
            )
            for loc in self.contact_locations
        ]

    def _build_graph(self):
        self.contact_vertices = [
            self.gcs.AddVertex(mode.get_convex_set(), mode.name)
            for mode in self.contact_modes
        ]

        if self.config.allow_teleportation:
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
            connections = list(combinations(range(self.num_contact_modes), 2))

            self.subgraphs = [
                self._build_subgraph_between_contact_modes(
                    mode_i, mode_j, self.config.no_cycles
                )
                for mode_i, mode_j in connections
            ]

            if self.config.use_entry_and_exit_subgraphs:
                self.source_subgraph = self._create_entry_or_exit_subgraph("entry")
                self.target_subgraph = self._create_entry_or_exit_subgraph("exit")

    def _build_subgraph_between_contact_modes(
        self,
        first_contact_mode_idx: int,
        second_contact_mode_idx: int,
        no_cycles: bool = False,
    ) -> NonCollisionSubGraph:
        subgraph = NonCollisionSubGraph.create_with_gcs(
            self.gcs,
            self.config,
            f"FACE_{first_contact_mode_idx}_to_FACE_{second_contact_mode_idx}",
        )
        if no_cycles:  # only connect lower idx faces to higher idx faces
            if first_contact_mode_idx <= second_contact_mode_idx:
                incoming_idx = first_contact_mode_idx
                outgoing_idx = second_contact_mode_idx
            else:  # second_contact_mode_idx <= first_contact_mode_idx
                outgoing_idx = first_contact_mode_idx
                incoming_idx = second_contact_mode_idx

            subgraph.connect_with_continuity_constraints(
                self.slider.geometry.get_collision_free_region_for_loc_idx(
                    incoming_idx
                ),
                VertexModePair(
                    self.contact_vertices[incoming_idx],
                    self.contact_modes[incoming_idx],
                ),
                incoming=True,
                outgoing=False,
            )
            subgraph.connect_with_continuity_constraints(
                self.slider.geometry.get_collision_free_region_for_loc_idx(
                    outgoing_idx
                ),
                VertexModePair(
                    self.contact_vertices[outgoing_idx],
                    self.contact_modes[outgoing_idx],
                ),
                incoming=False,
                outgoing=True,
            )
        else:
            for idx in (first_contact_mode_idx, second_contact_mode_idx):
                subgraph.connect_with_continuity_constraints(
                    self.slider.geometry.get_collision_free_region_for_loc_idx(idx),
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
        # Add all vertices from subgraphs
        if not self.config.allow_teleportation:
            for subgraph in self.subgraphs:
                all_pairs.update(subgraph.get_all_vertex_mode_pairs())

        # Add source and target vertices (and possibly the ones associated
        # with the entry and exit subgraphs)
        if (
            self.config.allow_teleportation
            or not self.config.use_entry_and_exit_subgraphs
        ):
            assert self.source is not None
            assert self.target is not None

            all_pairs[self.source.mode.name] = self.source
            all_pairs[self.target.mode.name] = self.target
        else:
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

        subgraph = NonCollisionSubGraph.create_with_gcs(self.gcs, self.config, name)

        for idx, (vertex, mode) in enumerate(
            zip(self.contact_vertices, self.contact_modes)
        ):
            subgraph.connect_with_continuity_constraints(
                self.slider.geometry.get_collision_free_region_for_loc_idx(idx),
                VertexModePair(vertex, mode),
                **kwargs,
            )
        return subgraph

    def set_initial_poses(
        self,
        finger_pose: PlanarPose,
        slider_pose: PlanarPose,
    ) -> None:
        self.finger_pose_initial = finger_pose
        self.slider_pose_initial = slider_pose

        if self.source is not None:
            print("Source vertex is already set, removing old vertex and adding new")
            self.gcs.RemoveVertex(self.source.vertex)
            self.source = None

        if (
            self.config.allow_teleportation
            or not self.config.use_entry_and_exit_subgraphs
        ):
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

        if self.target is not None:
            print("Target vertex is already set, removing old vertex and adding new")
            self.gcs.RemoveVertex(self.target.vertex)
            self.target = None

        if (
            self.config.allow_teleportation
            or not self.config.use_entry_and_exit_subgraphs
        ):
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
            self.config, slider_pose, finger_pose, initial_or_final
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

    def _solve(self, solver_params: PlanarSolverParams) -> MathematicalProgramResult:
        options = opt.GraphOfConvexSetsOptions()
        if solver_params.print_solver_output:
            options.solver_options = SolverOptions()
            # options.solver_options.SetOption(CommonSolverOption.kPrintFileName, "optimization_log.txt")  # type: ignore
            options.solver_options.SetOption(CommonSolverOption.kPrintToConsole, 1)  # type: ignore

        options.convex_relaxation = solver_params.gcs_convex_relaxation
        if options.convex_relaxation is True:
            options.preprocessing = True  # TODO(bernhardpg): should this be changed?
            options.max_rounded_paths = solver_params.gcs_max_rounded_paths

        options.solver = MosekSolver()

        assert self.source is not None
        assert self.target is not None

        result = self.gcs.SolveShortestPath(
            self.source.vertex, self.target.vertex, options
        )

        if solver_params.print_flows:
            self._print_edge_flows(result)

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
        self, solver_params: PlanarSolverParams
    ) -> (
        PlanarPushingTrajectory
        | Tuple[PlanarPushingTrajectory, PlanarPushingTrajectory]
    ):
        assert self.source is not None
        assert self.target is not None

        import time

        start = time.time()
        result = self._solve(solver_params)
        end = time.time()

        assert result.is_success()

        if solver_params.measure_solve_time:
            elapsed_time = end - start
            print(f"Total elapsed optimization time: {elapsed_time}")

        if solver_params.print_cost:
            cost = result.get_optimal_cost()
            print(f"Cost: {cost}")

        if solver_params.get_rounded_and_original_traj:
            original_traj = PlanarPushingTrajectory.from_result(
                self.config,
                result,
                self.gcs,
                self.source.vertex,
                self.target.vertex,
                self._get_all_vertex_mode_pairs(),
                False,
                solver_params.print_path,
                solver_params.assert_determinants,
            )
            rounded_traj = PlanarPushingTrajectory.from_result(
                self.config,
                result,
                self.gcs,
                self.source.vertex,
                self.target.vertex,
                self._get_all_vertex_mode_pairs(),
                True,
                False,  # don't need to print path twice
                solver_params.assert_determinants,
            )
            return original_traj, rounded_traj

        traj = PlanarPushingTrajectory.from_result(
            self.config,
            result,
            self.gcs,
            self.source.vertex,
            self.target.vertex,
            self._get_all_vertex_mode_pairs(),
            solver_params.nonlinear_traj_rounding,
            solver_params.print_path,
            solver_params.assert_determinants,
        )

        return traj

    def _print_edge_flows(self, result: MathematicalProgramResult) -> None:
        """
        Used for debugging.
        """
        edge_phis = {
            (e.u().name(), e.v().name()): result.GetSolution(e.phi())
            for e in self.gcs.Edges()
        }
        sorted_flows = sorted(edge_phis.items(), key=lambda item: item[0])
        for name, flow in sorted_flows:
            print(f"{name}: {flow}")

    def create_graph_diagram(
        self,
        filepath: Optional[Path] = None,
        result: Optional[MathematicalProgramResult] = None,
    ) -> pydot.Dot:
        """
        Optionally saves the graph to file if a string is given for the 'filepath' argument.
        """
        graphviz = self.gcs.GetGraphvizString(
            precision=2, result=result, show_slacks=False
        )

        data = pydot.graph_from_dot_data(graphviz)[0]  # type: ignore
        if filepath is not None:
            data.write_svg(str(filepath))

        return data
