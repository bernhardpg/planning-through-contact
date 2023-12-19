from typing import Dict, List, Optional

import numpy as np
import numpy.typing as npt
import pydrake.geometry.optimization as opt
import pydrake.symbolic as sym
from pydrake.solvers import (
    Binding,
    CommonSolverOption,
    Constraint,
    IpoptSolver,
    LinearConstraint,
    MathematicalProgram,
    MathematicalProgramResult,
    SnoptSolver,
    Solve,
    SolverOptions,
)

from planning_through_contact.geometry.planar.abstract_mode import (
    AbstractContactMode,
    AbstractModeVariables,
)
from planning_through_contact.geometry.planar.face_contact import FaceContactMode
from planning_through_contact.geometry.planar.non_collision_subgraph import (
    VertexModePair,
)
from planning_through_contact.geometry.planar.planar_pushing_trajectory import (
    PlanarPushingTrajectory,
)
from planning_through_contact.planning.planar.planar_plan_config import (
    PlanarPlanConfig,
    PlanarSolverParams,
)
from planning_through_contact.tools.gcs_tools import (
    get_gcs_solution_path_edges,
    get_gcs_solution_path_vertices,
)
from planning_through_contact.tools.types import NpVariableArray

GcsVertex = opt.GraphOfConvexSets.Vertex
GcsEdge = opt.GraphOfConvexSets.Edge


def assemble_progs_from_contact_modes(
    modes: List[AbstractContactMode], remove_redundant_constraints: bool = True
) -> MathematicalProgram:
    prog = MathematicalProgram()

    for mode in modes:
        mode_prog = mode.prog  # type: ignore
        if remove_redundant_constraints:
            if isinstance(mode, FaceContactMode):
                for c in mode.redundant_constraints:
                    mode_prog.RemoveConstraint(c)

        vars = mode_prog.decision_variables()
        prog.AddDecisionVariables(vars)

        for c in mode_prog.GetAllConstraints():
            prog.AddConstraint(c.evaluator(), c.variables())

        for c in mode_prog.GetAllCosts():
            prog.AddCost(c.evaluator(), c.variables())

    return prog


def get_mode_variables_from_constraint_variables(
    constraint_vars: NpVariableArray, pair_u: VertexModePair, pair_v: VertexModePair
) -> NpVariableArray:
    def _get_corresponding_pair(var: sym.Variable) -> VertexModePair:
        for pair in (pair_u, pair_v):
            if any([var.EqualTo(other) for other in pair.vertex.x()]):
                return pair
        # This should not exit without returning
        raise RuntimeError("Variable not found in any of the provided pairs")

    def _get_original_variable(var: sym.Variable) -> sym.Variable:
        pair = _get_corresponding_pair(var)
        # find indices by constructing a new mock program
        temp = MathematicalProgram()
        temp.AddDecisionVariables(pair.vertex.x())
        var_idx = temp.FindDecisionVariableIndex(var)

        if isinstance(pair.mode, FaceContactMode):
            prog = pair.mode.relaxed_prog
        else:  # NonCollisionMode
            prog = pair.mode.prog  # type: ignore
        return prog.decision_variables()[var_idx]  # type: ignore

    mode_vars = np.array([_get_original_variable(var) for var in constraint_vars])
    return mode_vars


def add_edge_constraints_to_prog(
    edges: List[GcsEdge], prog: MathematicalProgram, pairs: List[VertexModePair]
) -> None:
    for edge, (pair_u, pair_v) in zip(edges, zip(pairs[:-1], pairs[1:])):
        for c in edge.GetConstraints():
            vars = get_mode_variables_from_constraint_variables(
                c.variables(), pair_u, pair_v
            )
            prog.AddConstraint(c.evaluator(), vars)


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
        self.rounded_result = None
        self.config = pairs_on_path[0].mode.config

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

    def to_traj(
        self, solver_params: Optional[PlanarSolverParams] = None
    ) -> PlanarPushingTrajectory:
        if solver_params is not None and solver_params.nonlinear_traj_rounding:
            self.do_rounding(solver_params)
            return PlanarPushingTrajectory(self.config, self.get_rounded_vars())
        else:
            return PlanarPushingTrajectory(self.config, self.get_vars())

    def get_vars(self) -> List[AbstractModeVariables]:
        vars_on_path = [
            pair.mode.get_variable_solutions_for_vertex(pair.vertex, self.result)
            for pair in self.pairs
        ]
        return vars_on_path

    def do_rounding(self, solver_params: PlanarSolverParams) -> None:
        self.rounded_result = self._do_nonlinear_rounding(solver_params)

    def get_rounded_vars(self) -> List[AbstractModeVariables]:
        assert self.rounded_result is not None
        vars_on_path = [
            pair.mode.get_variable_solutions(self.rounded_result) for pair in self.pairs
        ]
        return vars_on_path

    def get_path_names(self) -> List[str]:
        names = [pair.vertex.name() for pair in self.pairs]
        return names

    def get_vertices(self) -> List[GcsVertex]:
        return [p.vertex for p in self.pairs]

    def _construct_nonlinear_program(self) -> MathematicalProgram:
        prog = assemble_progs_from_contact_modes([p.mode for p in self.pairs])
        add_edge_constraints_to_prog(self.edges, prog, self.pairs)
        return prog

    def _get_initial_guess(self) -> npt.NDArray[np.float64]:
        num_vars_in_modes = [p.mode.prog.num_vars() for p in self.pairs]
        all_vertex_vars_concatenated = np.concatenate(
            [
                pair.vertex.x()[:num_vars]
                for pair, num_vars in zip(self.pairs, num_vars_in_modes)
            ]
        )
        vertex_var_vals = self.result.GetSolution(all_vertex_vars_concatenated)
        return vertex_var_vals

    def _do_nonlinear_rounding(
        self,
        solver_params: PlanarSolverParams,
    ) -> MathematicalProgramResult:
        """
        Assembles one big nonlinear program and solves it with the SDP relaxation as the initial guess.

        NOTE: Ipopt does not work because of "too few degrees of freedom". Snopt must be used.
        """
        prog = self._construct_nonlinear_program()

        import time

        start = time.time()

        initial_guess = self._get_initial_guess()

        solver_options = SolverOptions()

        if solver_params.print_solver_output:
            # NOTE(bernhardpg): I don't think either SNOPT nor IPOPT supports this setting
            solver_options.SetOption(CommonSolverOption.kPrintToConsole, 1)  # type: ignore

        snopt = SnoptSolver()
        if solver_params.save_solver_output:
            solver_options.SetOption(
                snopt.solver_id(), "Print file", "snopt_output.txt"
            )

        solver_options.SetOption(
            snopt.solver_id(),
            "Major Feasibility Tolerance",
            solver_params.nonl_round_feas_tol,
        )
        solver_options.SetOption(
            snopt.solver_id(),
            "Minor Feasibility Tolerance",
            solver_params.nonl_round_feas_tol,
        )
        solver_options.SetOption(
            snopt.solver_id(),
            "Major Optimality Tolerance",
            solver_params.nonl_round_opt_tol,
        )
        # The performance seems to be better when this parameter is left to its default value
        # solver_options.SetOption(
        #     snopt.solver_id(),
        #     "Minor Optimality Tolerance",
        #     solver_params.nonl_round_opt_tol,
        # )
        solver_options.SetOption(
            snopt.solver_id(),
            "Major iterations limit",
            solver_params.nonl_round_major_iter_limit,
        )

        result = snopt.Solve(prog, initial_guess, solver_options=solver_options)  # type: ignore

        end = time.time()

        if solver_params.measure_solve_time:
            elapsed_time = end - start
            print(f"Total elapsed optimization time: {elapsed_time}")

        if solver_params.assert_rounding_res:
            if not result.is_success():
                print(
                    f"Solution was not successfull. Solution result: {result.get_solution_result()} "
                )
                raise RuntimeError("Rounding was not succesfull.")
        else:
            if not result.is_success():
                print("Warning! Rounding was not succesfull")

        return result
