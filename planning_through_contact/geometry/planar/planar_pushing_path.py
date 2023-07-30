from typing import Dict, List

import numpy as np
import pydrake.geometry.optimization as opt
import pydrake.symbolic as sym
from pydrake.solvers import (
    Binding,
    Constraint,
    LinearConstraint,
    MathematicalProgram,
    MathematicalProgramResult,
    Solve,
)

from planning_through_contact.geometry.planar.abstract_mode import AbstractModeVariables
from planning_through_contact.geometry.planar.non_collision_subgraph import (
    VertexModePair,
)
from planning_through_contact.tools.gcs_tools import (
    get_gcs_solution_path_edges,
    get_gcs_solution_path_vertices,
)
from planning_through_contact.tools.types import NpVariableArray

GcsVertex = opt.GraphOfConvexSets.Vertex
GcsEdge = opt.GraphOfConvexSets.Edge


class PlanarPushingPath:
    """
    Stores a sequence of contact modes of the type AbstractContactMode.
    """

    def __init__(
        self,
        pairs_on_path: List[VertexModePair],
        edges_on_path: List[GcsEdge],
        result: MathematicalProgramResult,
    ) -> None:
        self.pairs = pairs_on_path
        self.edges = edges_on_path
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
        vertex_path = get_gcs_solution_path_vertices(
            gcs, result, source_vertex, target_vertex, flow_treshold
        )
        edge_path = get_gcs_solution_path_edges(
            gcs, result, source_vertex, target_vertex, flow_treshold
        )
        pairs_on_path = [all_pairs[v.name()] for v in vertex_path]
        return cls(pairs_on_path, edge_path, result)

    def get_vars(self) -> List[AbstractModeVariables]:
        vars_on_path = [
            pair.mode.get_variable_solutions_for_vertex(pair.vertex, self.result)
            for pair in self.pairs
        ]
        return vars_on_path

    def get_rounded_vars(
        self, measure_time: bool = False
    ) -> List[AbstractModeVariables]:
        rounded_result = self._do_nonlinear_rounding(measure_time)
        vars_on_path = [
            pair.mode.get_variable_solutions(rounded_result) for pair in self.pairs
        ]
        return vars_on_path

    def get_path_names(self) -> List[str]:
        names = [pair.vertex.name() for pair in self.pairs]
        return names

    def get_vertices(self) -> List[GcsVertex]:
        return [p.vertex for p in self.pairs]

    # TODO(bernhardpg): move somewhere more suitable
    @staticmethod
    def get_mode_variables_from_constraint_variables(
        constraint_vars: NpVariableArray, pair_u: VertexModePair, pair_v: VertexModePair
    ) -> NpVariableArray:
        num_vars_u = pair_u.mode.prog.num_vars()
        num_vars_v = pair_v.mode.prog.num_vars()

        # don't add variables that are not in the modes (these come from the
        # SDP relaxation)
        vertex_vars_stacked = np.concatenate(
            (pair_u.vertex.x()[:num_vars_u], pair_v.vertex.x()[:num_vars_v])
        )

        # find indices by constructing a new mock program
        temp = MathematicalProgram()
        temp.AddDecisionVariables(vertex_vars_stacked)
        idxs = temp.FindDecisionVariableIndices(constraint_vars)

        mode_vars_stacked = np.concatenate(
            (
                pair_u.mode.prog.decision_variables(),
                pair_v.mode.prog.decision_variables(),
            )
        )
        return mode_vars_stacked[idxs]

    def _construct_nonlinear_program_from_path(self) -> MathematicalProgram:
        prog = MathematicalProgram()

        for p in self.pairs:
            mode_prog = p.mode.prog

            vars = mode_prog.decision_variables()
            prog.AddDecisionVariables(vars)

            for c in mode_prog.GetAllConstraints():
                prog.AddConstraint(c.evaluator(), c.variables())

            for c in mode_prog.GetAllCosts():
                prog.AddCost(c.evaluator(), c.variables())

        for edge, (pair_u, pair_v) in zip(
            self.edges, zip(self.pairs[:-1], self.pairs[1:])
        ):
            for c in edge.GetConstraints():
                vars = self.get_mode_variables_from_constraint_variables(
                    c.variables(), pair_u, pair_v
                )
                prog.AddConstraint(c.evaluator(), vars)

        return prog

    def _do_nonlinear_rounding(
        self, measure_time: bool = False
    ) -> MathematicalProgramResult:
        prog = self._construct_nonlinear_program_from_path()

        import time

        start = time.time()
        result = Solve(prog)
        end = time.time()

        if measure_time:
            elapsed_time = end - start
            print(f"Total elapsed optimization time: {elapsed_time}")

        return result
