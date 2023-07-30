from typing import Dict, List

import pydrake.geometry.optimization as opt
from pydrake.solvers import MathematicalProgram, MathematicalProgramResult

from planning_through_contact.geometry.planar.abstract_mode import AbstractModeVariables
from planning_through_contact.geometry.planar.non_collision_subgraph import (
    VertexModePair,
)
from planning_through_contact.tools.gcs_tools import get_gcs_solution_path

GcsVertex = opt.GraphOfConvexSets.Vertex
GcsEdge = opt.GraphOfConvexSets.Edge


class PlanarPushingPath:
    """
    Stores a sequence of contact modes of the type AbstractContactMode.
    """

    def __init__(
        self, pairs_on_path: List[VertexModePair], result: MathematicalProgramResult
    ) -> None:
        self.pairs = pairs_on_path
        self.result = result

    @classmethod
    def from_result(
        cls,
        gcs: opt.GraphOfConvexSets,
        result: MathematicalProgramResult,
        source_vertex: GcsVertex,
        target_vertex: GcsVertex,
        all_pairs: Dict[str, VertexModePair],
        flow_treshold: float = 0.55,
    ) -> "PlanarPushingPath":
        vertex_path = get_gcs_solution_path(
            gcs, result, source_vertex, target_vertex, flow_treshold
        )
        pairs_on_path = [all_pairs[v.name()] for v in vertex_path]
        return cls(pairs_on_path, result)

    def get_vars(self) -> List[AbstractModeVariables]:
        vars_on_path = [
            pair.mode.get_variable_solutions_for_vertex(pair.vertex, self.result)
            for pair in self.pairs
        ]
        return vars_on_path

    def get_path_names(self) -> List[str]:
        names = [pair.vertex.name() for pair in self.pairs]
        return names

    def get_vertices(self) -> List[GcsVertex]:
        return [p.vertex for p in self.pairs]

    def do_nonlinear_rounding(self) -> None:
        prog = MathematicalProgram()

        for p in self.pairs:
            mode_prog = p.mode.prog

            vars = mode_prog.decision_variables()
            breakpoint()
