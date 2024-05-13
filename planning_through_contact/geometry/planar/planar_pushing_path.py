from typing import Dict, List

import numpy as np
import numpy.typing as npt
import pydrake.geometry.optimization as opt
import pydrake.symbolic as sym
from pydrake.solvers import (
    CommonSolverOption,
    MathematicalProgram,
    MathematicalProgramResult,
    SnoptSolver,
    SolutionResult,
    SolverOptions,
)

from planning_through_contact.geometry.planar.abstract_mode import (
    AbstractContactMode,
    AbstractModeVariables,
)
from planning_through_contact.geometry.planar.face_contact import (
    FaceContactMode,
    FaceContactVariables,
)
from planning_through_contact.geometry.planar.non_collision import NonCollisionMode
from planning_through_contact.geometry.planar.non_collision_subgraph import (
    VertexModePair,
)
from planning_through_contact.geometry.planar.planar_pushing_trajectory import (
    PlanarPushingTrajectory,
)
from planning_through_contact.planning.planar.planar_plan_config import (
    PlanarSolverParams,
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

    @property
    def relaxed_cost(self) -> float:
        return self.result.get_optimal_cost()

    @property
    def rounded_cost(self) -> float:
        assert self.rounded_result is not None
        return self.rounded_result.get_optimal_cost()

    @property
    def solve_time(self) -> float:
        # The field `optimizer_time` is Mosek specific
        return self.result.get_solver_details().optimizer_time  # type: ignore

    # @property
    # def rounding_time(self) -> float:
    #     assert self.rounded_result is not None
    #     # The field `info` is Snopt specific
    #     info = self.rounded_result.get_solver_details().info
    #     # TODO(bernhardpg): How to get the SNOPT solver time?
    #     return None

    @classmethod
    def from_path(
        cls,
        gcs: opt.GraphOfConvexSets,
        result: MathematicalProgramResult,
        edge_path: List[GcsEdge],
        all_pairs: Dict[str, VertexModePair],
        assert_nan_values: bool = True,
    ) -> "PlanarPushingPath":
        """
        Creates a Planar Pushing path from a given set of edges.
        """
        vertex_path = [e.u() for e in edge_path]
        vertex_path.append(edge_path[-1].v())

        if assert_nan_values:
            # We need a result to do this
            assert result is not None

            def _check_all_nan_or_zero(array: npt.NDArray[np.float64]) -> bool:
                return np.isnan(array) | np.isclose(array, 0, atol=1e-5)

            # Assert that all decision varibles NOT ON the optimal path are NaN or 0
            vertices_not_on_path = [v for v in gcs.Vertices() if v not in vertex_path]
            if len(vertices_not_on_path) > 0:
                vertex_vars_not_on_path = np.concatenate(
                    [result.GetSolution(v.x()) for v in vertices_not_on_path]
                )
                assert np.all(_check_all_nan_or_zero(vertex_vars_not_on_path))

            # Assert that all decision varibles ON the optimal path are not NaN
            vertex_vars_on_path = np.concatenate(
                [result.GetSolution(v.x()) for v in vertex_path]
            )
            assert np.all(~np.isnan(vertex_vars_on_path))

        pairs_on_path = [all_pairs[v.name()] for v in vertex_path]
        return cls(pairs_on_path, edge_path, result)

    @classmethod
    def from_result(
        cls,
        gcs: opt.GraphOfConvexSets,
        result: MathematicalProgramResult,
        source_vertex: GcsVertex,
        target_vertex: GcsVertex,
        all_pairs: Dict[str, VertexModePair],
        assert_nan_values: bool = True,
    ) -> "PlanarPushingPath":
        """
        Create a PlanarPushingPath from a relaxed GCS result, which then picks
        a deterministic graph path using gcs.GetSolutionPath(...).
        """
        edge_path = gcs.GetSolutionPath(source_vertex, target_vertex, result)
        vertex_path = [e.u() for e in edge_path]
        vertex_path.append(edge_path[-1].v())

        if assert_nan_values:

            def _check_all_nan_or_zero(array: npt.NDArray[np.float64]) -> bool:
                return np.isnan(array) | np.isclose(array, 0, atol=1e-5)

            # Assert that all decision varibles NOT ON the optimal path are NaN or 0
            vertices_not_on_path = [v for v in gcs.Vertices() if v not in vertex_path]
            if len(vertices_not_on_path) > 0:
                vertex_vars_not_on_path = np.concatenate(
                    [result.GetSolution(v.x()) for v in vertices_not_on_path]
                )
                assert np.all(_check_all_nan_or_zero(vertex_vars_not_on_path))

            # Assert that all decision varibles ON the optimal path are not NaN
            vertex_vars_on_path = np.concatenate(
                [result.GetSolution(v.x()) for v in vertex_path]
            )
            assert np.all(~np.isnan(vertex_vars_on_path))

        pairs_on_path = [all_pairs[v.name()] for v in vertex_path]
        return cls(pairs_on_path, edge_path, result)

    def get_cost_terms(self) -> Dict[str, Dict]:
        contact_mode_costs = {
            "keypoint_arc": [],
            "keypoint_reg": [],
            "force_reg": [],
            "contact_time": [],
            "mode_cost": [],
        }

        contact_mode_sums = {
            "keypoint_arc": float,
            "keypoint_reg": float,
            "force_reg": float,
            "contact_time": float,
            "mode_cost": float,
        }

        non_collision_costs = {
            "pusher_arc_length": [],
            "pusher_vel_reg": [],
            "object_avoidance_socp": [],
            "non_contact_time": [],
        }

        non_collision_sums = {
            "pusher_arc_length": float,
            "pusher_vel_reg": float,
            "object_avoidance_socp": float,
            "non_contact_time": float,
        }

        def _get_cost_vals(mode, vertex, costs):
            cost_var_idxs = [
                mode.get_variable_indices_in_gcs_vertex(c.variables()) for c in costs
            ]
            cost_vars = [vertex.x()[idx] for idx in cost_var_idxs]
            cost_var_vals = [self.result.GetSolution(vars) for vars in cost_vars]
            cost_vals = [
                cost.evaluator().Eval(vals) for cost, vals in zip(costs, cost_var_vals)
            ]
            return cost_vals

        for key in contact_mode_costs.keys():
            for vertex, mode in self.pairs:
                if isinstance(mode, FaceContactMode):
                    contact_mode_costs[key].append(
                        np.sum(_get_cost_vals(mode, vertex, mode.costs[key]))
                    )

        for key in contact_mode_sums.keys():
            contact_mode_sums[key] = np.sum(contact_mode_costs[key])

        for key in non_collision_costs.keys():
            for vertex, mode in self.pairs:
                if isinstance(mode, NonCollisionMode):
                    non_collision_costs[key].append(
                        np.sum(_get_cost_vals(mode, vertex, mode.costs[key]))
                    )

        for key in non_collision_sums.keys():
            non_collision_sums[key] = np.sum(non_collision_costs[key])

        res = {
            "contact_mode_costs": contact_mode_costs,
            "contact_mode_sums": contact_mode_sums,
            "non_collision_costs": non_collision_costs,
            "non_collision_sums": non_collision_sums,
        }

        return res

    def to_traj(
        self,
        rounded: bool = False,
    ) -> PlanarPushingTrajectory:
        if rounded:
            assert self.rounded_result is not None
            return PlanarPushingTrajectory(
                self.config,
                self.get_rounded_vars(),
                self.rounded_result.get_optimal_cost(),
            )
        else:
            return PlanarPushingTrajectory(
                self.config, self.get_vars(), self.result.get_optimal_cost()
            )

    def get_vars(self) -> List[AbstractModeVariables]:
        vars_on_path = [
            pair.mode.get_variable_solutions_for_vertex(pair.vertex, self.result)
            for pair in self.pairs
        ]
        return vars_on_path

    def get_determinants(self, rounded: bool = False) -> List[float]:
        if rounded:
            face_contact_vars = [
                vars
                for vars in self.get_rounded_vars()
                if isinstance(vars, FaceContactVariables)
            ]
        else:  # rounded == False
            face_contact_vars = [
                vars
                for vars in self.get_vars()
                if isinstance(vars, FaceContactVariables)
            ]
        cos_ths = np.concatenate([var.cos_ths for var in face_contact_vars])
        sin_ths = np.concatenate([var.sin_ths for var in face_contact_vars])

        determinants: List[float] = [
            np.linalg.norm([cos, sin]) for cos, sin in zip(cos_ths, sin_ths)
        ]
        return determinants

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

    def _get_initial_guess_as_orig_variables(
        self, scale_rot_values: bool = True
    ) -> npt.NDArray[np.float64]:
        original_decision_var_idxs_in_vertices = [
            mode.get_variable_indices_in_gcs_vertex(mode.prog.decision_variables())
            for _, mode in self.pairs
        ]
        decision_vars_in_vertex_vars = [
            vertex.x()[idxs]
            for (vertex, _), idxs in zip(
                self.pairs, original_decision_var_idxs_in_vertices
            )
        ]
        all_vertex_vars_concatenated = np.concatenate(decision_vars_in_vertex_vars)

        vertex_var_vals = self.result.GetSolution(all_vertex_vars_concatenated)

        # This scales the rotational values so the initial guess starts with
        # (cos, sin) being on the unit circle
        if scale_rot_values:
            mock_prog = MathematicalProgram()
            mock_prog.AddDecisionVariables(all_vertex_vars_concatenated)

            def _scale_rot_vec(cos, sin):
                idx = mode.get_variable_indices_in_gcs_vertex(np.array([cos, sin]))
                vertex_vars = vertex.x()[idx]
                idx_in_stacked_vec = mock_prog.FindDecisionVariableIndices(vertex_vars)
                rot_vec = vertex_var_vals[idx_in_stacked_vec]
                length = np.linalg.norm(rot_vec)

                # Scale rotation parameters so they are on unit circle
                vertex_var_vals[idx_in_stacked_vec] = rot_vec / length

            for vertex, mode in self.pairs:
                if isinstance(mode, NonCollisionMode):
                    _scale_rot_vec(mode.variables.cos_th, mode.variables.sin_th)
                elif isinstance(mode, FaceContactMode):
                    for k in range(mode.num_knot_points):
                        _scale_rot_vec(
                            mode.variables.cos_ths[k], mode.variables.sin_ths[k]
                        )
                else:
                    raise NotImplementedError(f"Mode {type(mode)} not supported")

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

        initial_guess = self._get_initial_guess_as_orig_variables()

        solver_options = SolverOptions()

        if solver_params.print_solver_output:
            # NOTE(bernhardpg): I don't think either SNOPT nor IPOPT supports this setting
            solver_options.SetOption(CommonSolverOption.kPrintToConsole, 1)  # type: ignore

        snopt = SnoptSolver()
        if solver_params.save_solver_output:
            import os

            snopt_log_path = "snopt_output.txt"
            # Delete log file if it already exists as Snopt just keeps writing to the same file
            if os.path.exists(snopt_log_path):
                os.remove(snopt_log_path)

            solver_options.SetOption(snopt.solver_id(), "Print file", snopt_log_path)

        solver_options.SetOption(
            snopt.solver_id(),
            "Major Feasibility Tolerance",
            solver_params.nonl_round_major_feas_tol,
        )
        solver_options.SetOption(
            snopt.solver_id(),
            "Major Optimality Tolerance",
            solver_params.nonl_round_opt_tol,
        )
        # The performance seems to be better when these (minor step) parameters are left
        # to their default value
        # solver_options.SetOption(
        #     snopt.solver_id(),
        #     "Minor Feasibility Tolerance",
        #     solver_params.nonl_round_minor_feas_tol,
        # )
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

        start = time.time()
        result = snopt.Solve(prog, initial_guess, solver_options=solver_options)  # type: ignore
        end = time.time()
        self.rounding_time = end - start

        def _are_within_ranges_with_tolerance(
            values, lower_bounds, upper_bounds, tolerance
        ) -> npt.NDArray[np.bool_]:
            close_to_lower = np.isclose(values, lower_bounds, atol=tolerance)
            close_to_upper = np.isclose(values, upper_bounds, atol=tolerance)
            within_bounds = (values > lower_bounds) & (values < upper_bounds)
            return close_to_lower | close_to_upper | within_bounds

        # TODO(bernhardpg): Remove this, we don't use it.
        if False:
            if not result.is_success():
                # Sometimes SNOPT reports that it cannot proceed due to numerical errors, but the solution is still
                # feasible. In that case we keep it (empirically it is often close to optimal).
                if result.get_solution_result() == SolutionResult.kSolverSpecificError:
                    prog.SetInitialGuess(
                        prog.decision_variables(),
                        result.GetSolution(prog.decision_variables()),
                    )

                    TOL = solver_params.nonl_round_major_feas_tol

                    def _is_binding_satisfied(binding) -> npt.NDArray[np.bool_]:
                        val = prog.EvalBindingAtInitialGuess(binding)
                        ub, lb = (
                            binding.evaluator().upper_bound(),
                            binding.evaluator().lower_bound(),
                        )
                        return _are_within_ranges_with_tolerance(val, lb, ub, TOL)

                    constraints_satisfied = np.concatenate(
                        [
                            _is_binding_satisfied(binding)
                            for binding in prog.GetAllConstraints()
                        ]
                    )

                    # if result.get_optimal_cost() <= self.result.get_optimal_cost():
                    #     # This should not happen
                    #     calculated_cost = np.sum(
                    #         [prog.EvalBindingAtInitialGuess(b) for b in prog.GetAllCosts()]
                    #     )
                    #     if not np.isclose(result.get_optimal_cost(), calculated_cost):
                    #         breakpoint()

                    cost_upper_bound = (
                        result.get_optimal_cost() >= self.result.get_optimal_cost()
                    )
                    solution_feasible = np.all(constraints_satisfied)
                    if solution_feasible and cost_upper_bound:
                        result.set_solution_result(SolutionResult.kSolutionFound)

        if solver_params.assert_rounding_res:
            if not result.is_success():
                print(
                    f"Solution was not successfull. Solution result: {result.get_solution_result()} "
                )
                raise RuntimeError("Rounding was not succesfull.")

        return result
