from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Literal, NamedTuple, Tuple

import pydrake.geometry.optimization as opt
import pydrake.symbolic as sym
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
from planning_through_contact.geometry.planar.planar_contact_modes import (
    AbstractModeVariables,
    FaceContactMode,
    FaceContactVariables,
    NonCollisionMode,
    NonCollisionVariables,
    PlanarPlanSpecs,
)
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.planar.trajectory_builder import (
    PlanarTrajectory,
    PlanarTrajectoryBuilder,
)
from planning_through_contact.geometry.rigid_body import RigidBody

GcsVertex = opt.GraphOfConvexSets.Vertex
GcsEdge = opt.GraphOfConvexSets.Edge
BidirGcsEdge = Tuple[GcsEdge, GcsEdge]


class VertexModePair(NamedTuple):
    vertex: GcsVertex
    mode: FaceContactMode | NonCollisionMode


@dataclass
class NonCollisionSubGraph:
    sets: List[opt.ConvexSet]
    modes: List[NonCollisionMode]
    vertices: List[GcsVertex]
    edges: Dict[Tuple[int, int], BidirGcsEdge]
    graph_connections: Dict[int, BidirGcsEdge]

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
        as well as the modes. The modes each has the option to get a convex set from its underlying mathematical program,
        which is added as vertices to the GCS instance.

        An edge is added between any two overlapping position modes, as well as between the incoming and outgoing
        nodes to the bigger graph.

        @param mode_i: Index of first contact mode where this subgraph is connected
        @param mode_j: Index of second contact mode where this subgraph is connected
        """

        sets = [mode.get_convex_set() for mode in modes]

        vertex_names = [f"{mode_i}_TO_{mode_j}_{mode.name}" for mode in modes]
        vertices = [gcs.AddVertex(s, name) for s, name in zip(sets, vertex_names)]

        edge_idxs = cls._get_edge_idxs(modes)
        # Add bi-directional edges
        edges = {
            (i, j): (
                gcs.AddEdge(vertices[i], vertices[j]),
                gcs.AddEdge(vertices[j], vertices[i]),
            )
            for i, j in edge_idxs
        }

        return cls(sets, modes, vertices, edges, {})

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

    def __post_init__(self) -> None:
        for (i, j), (first_edge, second_edge) in self.edges.items():
            # Bi-directional edges
            self._add_continuity_constraints(i, j, first_edge)
            self._add_continuity_constraints(j, i, second_edge)

    def _add_continuity_constraints(
        self, outgoing_idx: int, incoming_idx: int, edge: GcsEdge
    ):
        first_vars = self.modes[incoming_idx].get_continuity_vars("first").vector
        first_var_idxs = self.modes[incoming_idx].get_variable_indices_in_gcs_vertex(
            first_vars
        )

        last_vars = self.modes[outgoing_idx].get_continuity_vars("last").vector
        last_var_idxs = self.modes[outgoing_idx].get_variable_indices_in_gcs_vertex(
            last_vars
        )

        constraint = eq(edge.xu()[last_var_idxs], edge.xv()[first_var_idxs])
        for c in constraint:
            edge.AddConstraint(c)

    def add_connection_to_full_graph(
        self,
        gcs: opt.GraphOfConvexSets,
        connection_vertex: GcsVertex,
        connection_idx: int,
    ) -> None:
        """
        Adds a bi-directional edge between the provided connection vertex and the subgraph vertex with index connection_idx in this subgraph.
        """

        self.graph_connections[connection_idx] = (
            gcs.AddEdge(connection_vertex, self.vertices[connection_idx]),
            gcs.AddEdge(self.vertices[connection_idx], connection_vertex),
        )

    def get_all_vertex_mode_pairs(self) -> Dict[str, VertexModePair]:
        return {
            v.name(): VertexModePair(vertex=v, mode=m)
            for v, m in zip(self.vertices, self.modes)
        }


def _find_path_to_target(
    edges: List[opt.GraphOfConvexSets.Edge],
    target: opt.GraphOfConvexSets.Vertex,
    u: opt.GraphOfConvexSets.Vertex,
) -> List[opt.GraphOfConvexSets.Vertex]:
    current_edge = next(e for e in edges if e.u() == u)
    v = current_edge.v()
    target_reached = v == target
    if target_reached:
        return [u] + [v]
    else:
        return [u] + _find_path_to_target(edges, target, v)


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
        self._add_costs()
        self._add_continuity_constraints_for_transitions()
        self._collect_all_vertex_mode_pairs()

    @property
    def num_contact_modes(self) -> int:
        return len(self.contact_modes)

    def _formulate_contact_modes(self):
        contact_locations = self.slider.geometry.contact_locations
        # TODO: should just extract faces, rather than relying on the object to only pass faces as
        # contact locations

        if not all([loc.pos == ContactLocation.FACE for loc in contact_locations]):
            raise RuntimeError("Only face contacts are supported for planar pushing.")

        self.contact_modes = [
            FaceContactMode.create_from_plan_spec(loc, self.plan_specs, self.slider)
            for loc in contact_locations
        ]

        self.non_collision_modes = [
            NonCollisionMode.create_from_plan_spec(loc, self.plan_specs, self.slider)
            for loc in contact_locations
        ]

    def _build_graph(self):
        self.contact_vertices = [
            self.gcs.AddVertex(mode.get_convex_set(), mode.name)
            for mode in self.contact_modes
        ]

        self.subgraphs = [
            self._build_subgraph(mode_i, mode_j)
            for mode_i, mode_j in combinations(range(self.num_contact_modes), 2)
        ]

    def _add_costs(self):
        # Contact modes
        for mode, vertex in zip(self.contact_modes, self.contact_vertices):
            var_idxs, evaluators = mode.get_cost_terms()
            vars = vertex.x()[var_idxs]
            bindings = [Binding[LinearCost](e, v) for e, v in zip(evaluators, vars)]
            for b in bindings:
                vertex.AddCost(b)

        # Non collision modes
        for subgraph in self.subgraphs:
            for mode, vertex in zip(subgraph.modes, subgraph.vertices):
                var_idxs, evaluator = mode.get_cost_term()
                vars = vertex.x()[var_idxs]
                binding = Binding[QuadraticCost](evaluator, vars)
                vertex.AddCost(binding)

    def _build_subgraph(self, mode_i: int, mode_j: int) -> NonCollisionSubGraph:
        subgraph = NonCollisionSubGraph.from_modes(
            self.non_collision_modes, self.gcs, mode_i, mode_j
        )
        subgraph.add_connection_to_full_graph(
            self.gcs, self.contact_vertices[mode_i], mode_i
        )
        subgraph.add_connection_to_full_graph(
            self.gcs, self.contact_vertices[mode_j], mode_j
        )
        return subgraph

    def _add_continuity_constraints_for_transitions(self) -> None:
        for subgraph in self.subgraphs:
            for mode_idx, (
                first_edge,
                second_edge,
            ) in subgraph.graph_connections.items():
                # bi-directional edges
                self._add_cont_const_btwn_contact_and_non_collision(
                    self.contact_modes[mode_idx],
                    subgraph.modes[mode_idx],
                    first_edge,  # from contact mode to non-collision
                    "outgoing",
                )
                self._add_cont_const_btwn_contact_and_non_collision(
                    self.contact_modes[mode_idx],
                    subgraph.modes[mode_idx],
                    second_edge,  # from non-collision to contact mode
                    "incoming",
                )

    @staticmethod
    def _add_cont_const_btwn_contact_and_non_collision(
        contact_mode: FaceContactMode,
        non_collision_mode: NonCollisionMode,
        edge: GcsEdge,
        incoming_or_outgoing: Literal["incoming", "outgoing"],
    ):
        """
        @param incoming_or_outgoing: Whether the edge is from a contact mode to a non_collision mode, or the other way around:
            "outgoing": contact_mode -> non_collision_mode
            "incoming": contact_mode <- non_collision_mode
        """
        if incoming_or_outgoing == "incoming":
            non_collision_vars_last = non_collision_mode.get_continuity_vars(
                "last"
            ).vector
            last_var_idxs = non_collision_mode.get_variable_indices_in_gcs_vertex(
                non_collision_vars_last
            )
            lhs = edge.xu()[last_var_idxs]

            contact_vars_first = contact_mode.get_continuity_vars("first")
            A, b = sym.DecomposeAffineExpressions(
                contact_vars_first.vector, contact_vars_first.get_pure_variables()
            )

            first_var_idxs = contact_mode.get_variable_indices_in_gcs_vertex(
                contact_vars_first.get_pure_variables()
            )

            rhs = A.dot(edge.xv()[first_var_idxs]) + b

        else:
            contact_vars_last = contact_mode.get_continuity_vars("last")
            A, b = sym.DecomposeAffineExpressions(
                contact_vars_last.vector, contact_vars_last.get_pure_variables()
            )

            last_var_idxs = contact_mode.get_variable_indices_in_gcs_vertex(
                contact_vars_last.get_pure_variables()
            )

            lhs = A.dot(edge.xu()[last_var_idxs]) + b

            non_collision_vars_first = non_collision_mode.get_continuity_vars(
                "first"
            ).vector
            first_var_idxs = non_collision_mode.get_variable_indices_in_gcs_vertex(
                non_collision_vars_first
            )
            rhs = edge.xv()[first_var_idxs]

        constraint = eq(lhs, rhs)
        for c in constraint:
            edge.AddConstraint(c)

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

    def _get_path(
        self,
        result: MathematicalProgramResult,
        flow_treshold: float = 0.55,
        print_path: bool = False,
    ) -> List[FaceContactVariables | NonCollisionVariables]:
        flow_variables = [e.phi() for e in self.gcs.Edges()]
        flow_results = [result.GetSolution(p) for p in flow_variables]
        active_edges = [
            edge
            for edge, flow in zip(self.gcs.Edges(), flow_results)
            if flow >= flow_treshold
        ]
        vertex_path = _find_path_to_target(
            active_edges, self.target_vertex, self.source_vertex
        )
        pairs_on_path = [
            self.all_pairs[v.name()]
            for v in vertex_path
            if v.name() not in ["source", "target"]
        ]
        full_path = [
            pair.mode.get_variable_solutions(pair.vertex, result)
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

        path = self._get_path(result, print_path=print_path)
        traj = PlanarTrajectoryBuilder(path).get_trajectory(interpolate=interpolate)

        return traj

    def save_graph_diagram(self, filepath: Path) -> None:
        graphviz = self.gcs.GetGraphvizString()
        import pydot

        data = pydot.graph_from_dot_data(graphviz)[0]  # type: ignore
        data.write_svg(str(filepath))
