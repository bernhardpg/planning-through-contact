from itertools import combinations
from typing import Dict, List, Literal, Optional, Tuple

import pydot
import pydrake.geometry.optimization as opt
from pydrake.geometry.optimization import Point
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
    ContactCostType,
    PlanarPlanConfig,
    PlanarSolverParams,
)

GcsVertex = opt.GraphOfConvexSets.Vertex
GcsEdge = opt.GraphOfConvexSets.Edge
BidirGcsEdge = Tuple[GcsEdge, GcsEdge]


class PlanarPushingPlanner:
    """
    A planner that generates motion plans for pushing an object (the "slider") with a point finger (the "pusher").
    The motion planner formulates the problem as a Graph-of-Convex-Sets problem, where each vertex in the graph
    corresponds to a contact mode.
    """

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

    def formulate_problem(self) -> None:
        assert self.config.start_and_goal is not None
        self.slider_pose_initial = self.config.start_and_goal.slider_initial_pose
        self.slider_pose_target = self.config.start_and_goal.slider_target_pose
        self.pusher_pose_initial = self.config.start_and_goal.pusher_initial_pose
        self.pusher_pose_target = self.config.start_and_goal.pusher_target_pose

        self.gcs = opt.GraphOfConvexSets()
        self._formulate_contact_modes()
        self._build_graph()

        # costs for non-collisions are added by each of the separate subgraphs
        for m, v in zip(self.contact_modes, self.contact_vertices):
            m.add_cost_to_vertex(v)

        if self.config.contact_config.mode_transition_cost is not None:
            for v in self.contact_vertices:
                v.AddCost(self.config.contact_config.mode_transition_cost)

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

        for mode in self.contact_modes:
            mode.add_so2_cut(
                self.slider_pose_initial.theta, self.slider_pose_target.theta
            )

    def _build_graph(self):
        self.contact_vertices = [
            self.gcs.AddVertex(mode.get_convex_set(), mode.name)
            for mode in self.contact_modes
        ]

        self.edges = {}
        if self.config.allow_teleportation:
            for i, j in combinations(range(self.num_contact_modes), 2):
                self.edges[
                    (self.contact_modes[i].name, self.contact_modes[j].name)
                ] = gcs_add_edge_with_continuity(
                    self.gcs,
                    VertexModePair(self.contact_vertices[i], self.contact_modes[i]),
                    VertexModePair(self.contact_vertices[j], self.contact_modes[j]),
                    only_continuity_on_slider=True,
                )
                self.edges[
                    (self.contact_modes[j].name, self.contact_modes[i].name)
                ] = gcs_add_edge_with_continuity(
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

        self._set_initial_poses(self.pusher_pose_initial, self.slider_pose_initial)
        self._set_target_poses(self.pusher_pose_target, self.slider_pose_target)

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

    def _set_initial_poses(
        self,
        pusher_pose: PlanarPose,
        slider_pose: PlanarPose,
    ) -> None:
        if (
            self.config.allow_teleportation
            or not self.config.use_entry_and_exit_subgraphs
        ):
            self.source = self._add_single_source_or_target(
                pusher_pose, slider_pose, "initial"
            )
        else:
            self.source_subgraph.set_initial_poses(pusher_pose, slider_pose)
            self.source = self.source_subgraph.source

    def _set_target_poses(
        self,
        pusher_pose: PlanarPose,
        slider_pose: PlanarPose,
    ) -> None:
        if (
            self.config.allow_teleportation
            or not self.config.use_entry_and_exit_subgraphs
        ):
            self.target = self._add_single_source_or_target(
                pusher_pose, slider_pose, "final"
            )
        else:
            self.target_subgraph.set_final_poses(pusher_pose, slider_pose)
            self.target = self.target_subgraph.target

    def _add_single_source_or_target(
        self,
        pusher_pose: PlanarPose,
        slider_pose: PlanarPose,
        initial_or_final: Literal["initial", "final"],
    ) -> VertexModePair:
        if (
            initial_or_final == "final"
            and self.config.contact_config.cost_type == ContactCostType.OPTIMAL_CONTROL
        ):  # we don't enforce target position for slider with this cost
            # set_slider_pose = False
            # terminal_cost = True
            set_slider_pose = True
            terminal_cost = False
        else:
            set_slider_pose = True
            terminal_cost = False

        mode = NonCollisionMode.create_source_or_target_mode(
            self.config,
            slider_pose,
            pusher_pose,
            initial_or_final,
            set_slider_pose=set_slider_pose,
            terminal_cost=terminal_cost,
        )
        vertex = self.gcs.AddVertex(mode.get_convex_set(), mode.name)
        pair = VertexModePair(vertex, mode)

        if terminal_cost:  # add cost on target vertex
            mode.add_cost_to_vertex(vertex)

        # connect source or target to all contact modes
        if initial_or_final == "initial":
            # source to contact modes
            for contact_vertex, contact_mode in zip(
                self.contact_vertices, self.contact_modes
            ):
                self.edges[
                    ("source", contact_mode.name)
                ] = gcs_add_edge_with_continuity(
                    self.gcs,
                    pair,
                    VertexModePair(contact_vertex, contact_mode),
                    only_continuity_on_slider=True,
                )
        else:  # contact modes to target
            for contact_vertex, contact_mode in zip(
                self.contact_vertices, self.contact_modes
            ):
                self.edges[
                    (contact_mode.name, "target")
                ] = gcs_add_edge_with_continuity(
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

        if solver_params.save_solver_output:
            options.solver_options = SolverOptions()
            options.solver_options.SetOption(CommonSolverOption.kPrintFileName, "solver_log.txt")  # type: ignore

        options.convex_relaxation = solver_params.gcs_convex_relaxation
        if options.convex_relaxation is True:
            options.preprocessing = True  # TODO(bernhardpg): should this be changed?
            options.max_rounded_paths = solver_params.gcs_max_rounded_paths

        mosek = MosekSolver()
        options.solver = mosek
        options.solver_options.SetOption(
            mosek.solver_id(), "MSK_DPAR_INTPNT_CO_TOL_PFEAS", 1e-3
        )
        options.solver_options.SetOption(
            mosek.solver_id(), "MSK_DPAR_INTPNT_CO_TOL_DFEAS", 1e-3
        )
        options.solver_options.SetOption(
            mosek.solver_id(), "MSK_DPAR_INTPNT_CO_TOL_REL_GAP", 1e-3
        )

        assert self.source is not None
        assert self.target is not None

        # TODO: The following commented out code allows you to pick which path to choose
        # active_vertices = ["source", "FACE_2", "FACE_0", "target"]
        # active_edges = [
        #     self.edges[(active_vertices[i], active_vertices[i + 1])]
        #     for i in range(len(active_vertices) - 1)
        # ]
        # result = self.gcs.SolveConvexRestriction(active_edges, options)

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

        if solver_params.assert_result:
            assert result.is_success()
        else:
            if not result.is_success():
                print("WARNING: Solver did not find a solution!")

        if solver_params.measure_solve_time:
            elapsed_time = end - start
            print(f"Total elapsed optimization time: {elapsed_time}")

        if solver_params.print_cost:
            cost = result.get_optimal_cost()
            print(f"Cost: {cost}")

        self.path = PlanarPushingPath.from_result(
            self.gcs,
            result,
            self.source.vertex,
            self.target.vertex,
            self._get_all_vertex_mode_pairs(),
        )
        if solver_params.print_path:
            print(f"path: {self.path.get_path_names()}")

        if solver_params.nonlinear_traj_rounding:
            raise NotImplementedError("Not implemented yet")

        return PlanarPushingTrajectory(self.config, self.path.get_vars())

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
        filename: Optional[str] = None,
        result: Optional[MathematicalProgramResult] = None,
    ) -> pydot.Dot:
        """
        Optionally saves the graph to file if a string is given for the 'filepath' argument.
        """
        graphviz = self.gcs.GetGraphvizString(
            precision=2, result=result, show_slacks=False
        )

        data = pydot.graph_from_dot_data(graphviz)[0]  # type: ignore
        if filename is not None:
            data.write_png(filename + ".png")

        return data
