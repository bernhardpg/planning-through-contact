from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import numpy.typing as npt
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

from geometry.collision_geometry.collision_geometry import (
    ContactLocation,
    PolytopeContactLocation,
)
from geometry.planar.planar_contact_modes import (
    AbstractContactMode,
    FaceContactMode,
    NonCollisionMode,
    PlanarPlanSpecs,
)
from geometry.planar.planar_pose import PlanarPose
from geometry.rigid_body import RigidBody

GcsVertex = opt.GraphOfConvexSets.Vertex
GcsEdge = opt.GraphOfConvexSets.Edge
BidirGcsEdge = Tuple[GcsEdge, GcsEdge]


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
        and adds the vertices and edges to the GCS instance.

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
        last_vars = self.modes[outgoing_idx].get_continuity_vars("last").vector

        first_var_idxs = self.modes[incoming_idx].prog.FindDecisionVariableIndices(
            first_vars
        )
        last_var_idxs = self.modes[outgoing_idx].prog.FindDecisionVariableIndices(
            last_vars
        )

        constraint = eq(edge.xu()[first_var_idxs], edge.xv()[last_var_idxs])
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
        # self._add_continuity_constraints()

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

    def _add_continuity_constraints(self) -> None:
        for subgraph in self.subgraphs:
            # connection between contact modes and subgraph
            for contact_mode_idx, edges in subgraph.graph_connections.items():
                self._add_continuity_constraint(
                    self.contact_modes[contact_mode_idx],
                    subgraph.modes[contact_mode_idx],
                    edges,
                )

    # @staticmethod
    # def _add_continuity_constraint(
    #     mode_i: AbstractContactMode,
    #     mode_j: AbstractContactMode,
    #     edges: List[GcsEdge] | BidirGcsEdge,
    # ) -> None:
    # first_vars = mode_i.get_first_continuity_vars()
    # last_vars = mode_j.get_last_continuity_vars()
    # first_vars.get_variables()
    #
    # constraint = eq(first_vars.vector, last_vars.vector)
    # for c in constraint:
    #     breakpoint()
    #
    # for edge in edges:
    #     for c in cont_constraint:
    #         edge.AddConstraint(c)
    # breakpoint()

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
        options = opt.GraphOfConvexSetsOptions()
        options.convex_relaxation = True
        options.solver_options = SolverOptions()
        options.solver_options.SetOption(CommonSolverOption.kPrintToConsole, 1)  # type: ignore

        if options.convex_relaxation is True:
            options.preprocessing = True  # TODO Do I need to deal with this?
            options.max_rounded_paths = 1

        result = self.gcs.SolveShortestPath(
            self.source_vertex, self.target_vertex, options
        )
        return result

    def save_graph_diagram(self, filepath: Path) -> None:
        graphviz = self.gcs.GetGraphvizString()
        import pydot

        data = pydot.graph_from_dot_data(graphviz)[0]  # type: ignore
        data.write_svg(str(filepath))
