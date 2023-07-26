from itertools import combinations
from pathlib import Path
from typing import List, Tuple

import pydrake.geometry.optimization as opt
from pydrake.math import eq
from pydrake.solvers import (
    Binding,
    CommonSolverOption,
    LinearCost,
    MathematicalProgramResult,
    QuadraticCost,
    SolverOptions,
)

from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    ContactLocation,
)
from planning_through_contact.geometry.planar.face_contact import (
    FaceContactMode,
    FaceContactVariables,
)
from planning_through_contact.geometry.planar.non_collision import NonCollisionVariables
from planning_through_contact.geometry.planar.non_collision_subgraph import (
    NonCollisionSubGraph,
    VertexModePair,
)
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.planar.trajectory_builder import (
    PlanarTrajectory,
    PlanarTrajectoryBuilder,
)
from planning_through_contact.geometry.rigid_body import RigidBody
from planning_through_contact.planning.planar.planar_plan_specs import PlanarPlanSpecs
from planning_through_contact.tools.gcs_tools import get_gcs_solution_path

GcsVertex = opt.GraphOfConvexSets.Vertex
GcsEdge = opt.GraphOfConvexSets.Edge
BidirGcsEdge = Tuple[GcsEdge, GcsEdge]


class PlanarPushingPlanner:
    """
    A planner that generates motion plans for pushing an object (the "slider") with a point finger (the "pusher").
    The motion planner formulates the problem as a Graph-of-Convex-Sets problem, where each vertex in the graph
    corresponds to a contact mode.
    """

    def __init__(self, slider: RigidBody, plan_specs: PlanarPlanSpecs):
        self.slider = slider
        self.plan_specs = plan_specs

        self.gcs = opt.GraphOfConvexSets()
        self._formulate_contact_modes()
        self._build_graph()
        # costs for non-collisions are added by each of the separate subgraphs
        self._add_contact_mode_costs()
        self._collect_all_vertex_mode_pairs()

    @property
    def num_contact_modes(self) -> int:
        return len(self.contact_modes)

    def _formulate_contact_modes(self):
        # TODO(bernhardpg): should just extract faces, rather than relying on the
        # object to only pass faces as contact locations
        contact_locations = self.slider.geometry.contact_locations

        if not all([loc.pos == ContactLocation.FACE for loc in contact_locations]):
            raise RuntimeError("Only face contacts are supported for planar pushing.")

        self.contact_modes = [
            FaceContactMode.create_from_plan_spec(loc, self.plan_specs, self.slider)
            for loc in contact_locations
        ]

    def _build_graph(self):
        self.contact_vertices = [
            self.gcs.AddVertex(mode.get_convex_set(), mode.name)
            for mode in self.contact_modes
        ]

        self.subgraphs = [
            self._build_subgraph_between_contact_modes(mode_i, mode_j)
            for mode_i, mode_j in combinations(range(self.num_contact_modes), 2)
        ]

    def _add_contact_mode_costs(self):
        for mode, vertex in zip(self.contact_modes, self.contact_vertices):
            var_idxs, evaluators = mode.get_cost_terms()
            vars = vertex.x()[var_idxs]
            bindings = [Binding[LinearCost](e, v) for e, v in zip(evaluators, vars)]
            for b in bindings:
                vertex.AddCost(b)

    def _build_subgraph_between_contact_modes(
        self, first_contact_mode_idx: int, second_contact_mode_idx: int
    ) -> NonCollisionSubGraph:
        subgraph = NonCollisionSubGraph.create_with_gcs(
            self.gcs,
            self.slider,
            self.plan_specs,
            f"FACE_{first_contact_mode_idx}_to_FACE_{second_contact_mode_idx}",
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

    def _collect_all_vertex_mode_pairs(self) -> None:
        all_pairs = {
            v.name(): VertexModePair(vertex=v, mode=m)
            for v, m in zip(self.contact_vertices, self.contact_modes)
        }
        for subgraph in self.subgraphs:
            all_pairs.update(subgraph.get_all_vertex_mode_pairs())

        self.all_pairs = all_pairs

    def set_pusher_initial_pose(
        self, pose: PlanarPose, disregard_rotation: bool = True
    ) -> None:
        raise NotImplementedError("Setting the pose of the pusher is not yet supported")

    def set_pusher_final_pose(
        self, pose: PlanarPose, disregard_rotation: bool = True
    ) -> None:
        raise NotImplementedError("Setting the pose of the pusher is not yet supported")

    def set_slider_initial_pose(self, pose: PlanarPose) -> None:
        point = opt.Point(pose.full_vector())
        self.source_vertex = self.gcs.AddVertex(point, name="source")
        self.source_edges = [
            self.gcs.AddEdge(self.source_vertex, v) for v in self.contact_vertices
        ]
        self._add_continuity_constraint_with_source()
        # TODO: Cartesian product between slider and pusher initial pose

    def _add_continuity_constraint_with_source(self) -> None:
        # TODO(bernhardpg): remove
        for edge, mode in zip(self.source_edges, self.contact_modes):
            source_vars = edge.xu()

            # TODO: also incorporate continuity constraints on the finger
            first_vars = mode.get_continuity_vars("first").vector[2:]
            first_var_idxs = mode.get_variable_indices_in_gcs_vertex(first_vars)

            constraint = eq(source_vars, edge.xv()[first_var_idxs])
            for c in constraint:
                edge.AddConstraint(c)

    def set_slider_target_pose(self, pose: PlanarPose) -> None:
        point = opt.Point(pose.full_vector())
        self.target_vertex = self.gcs.AddVertex(point, name="target")
        self.target_edges = [
            self.gcs.AddEdge(v, self.target_vertex) for v in self.contact_vertices
        ]
        self._add_continuity_constraint_with_target()
        # TODO: Cartesian product between slider and pusher target pose

    def _add_continuity_constraint_with_target(self) -> None:
        for edge, mode in zip(self.target_edges, self.contact_modes):
            target_vars = edge.xv()

            # TODO: also incorporate continuity constraints on the finger
            last_vars = mode.get_continuity_vars("last").vector[2:]
            last_var_idxs = mode.get_variable_indices_in_gcs_vertex(last_vars)

            constraint = eq(edge.xu()[last_var_idxs], target_vars)
            for c in constraint:
                edge.AddConstraint(c)

    def _solve(self, print_output: bool = False) -> MathematicalProgramResult:
        options = opt.GraphOfConvexSetsOptions()
        options.convex_relaxation = True
        if print_output:
            options.solver_options = SolverOptions()
            options.solver_options.SetOption(CommonSolverOption.kPrintToConsole, 1)  # type: ignore

        if options.convex_relaxation is True:
            options.preprocessing = True  # TODO Do I need to deal with this?
            options.max_rounded_paths = 1

        result = self.gcs.SolveShortestPath(
            self.source_vertex, self.target_vertex, options
        )
        return result

    def _get_gcs_solution_path(
        self,
        result: MathematicalProgramResult,
        flow_treshold: float = 0.55,
        print_path: bool = False,
    ) -> List[FaceContactVariables | NonCollisionVariables]:
        vertex_path = get_gcs_solution_path(
            self.gcs, result, self.source_vertex, self.target_vertex, flow_treshold
        )
        pairs_on_path = [
            self.all_pairs[v.name()]
            for v in vertex_path
            if v.name() not in ["source", "target"]
        ]
        full_path = [
            pair.mode.get_variable_solutions_for_vertex(pair.vertex, result)
            for pair in pairs_on_path
        ]

        if print_path:
            names = [v.name() for v in vertex_path]
            print("Vertices on path:")
            for name in names:
                print(f" - {name}")

        return full_path

    def make_trajectory(
        self,
        print_path: bool = False,
        print_output: bool = False,
        measure_time: bool = False,
        interpolate: bool = True,
    ) -> PlanarTrajectory:
        import time

        start = time.time()
        result = self._solve(print_output)
        assert result.is_success()
        end = time.time()

        if measure_time:
            elapsed_time = end - start
            print(f"Total elapsed optimization time: {elapsed_time}")

        path = self._get_gcs_solution_path(result, print_path=print_path)
        traj = PlanarTrajectoryBuilder(path).get_trajectory(interpolate=interpolate)

        return traj

    def save_graph_diagram(self, filepath: Path) -> None:
        graphviz = self.gcs.GetGraphvizString()
        import pydot

        data = pydot.graph_from_dot_data(graphviz)[0]  # type: ignore
        data.write_svg(str(filepath))
