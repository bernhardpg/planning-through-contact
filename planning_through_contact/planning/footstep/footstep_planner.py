from typing import Dict, List, Literal, NamedTuple, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pydot
from pydrake.geometry.optimization import (
    GraphOfConvexSets,
    GraphOfConvexSetsOptions,
    Point,
)
from pydrake.math import eq, ge, le
from pydrake.solvers import (
    Binding,
    CommonSolverOption,
    MathematicalProgram,
    MathematicalProgramResult,
    SnoptSolver,
    SolutionResult,
    Solve,
    SolverOptions,
)
from pydrake.symbolic import Variable, Variables
from tqdm import tqdm

from planning_through_contact.planning.footstep.footstep_plan_config import (
    FootstepPlanningConfig,
)
from planning_through_contact.planning.footstep.footstep_trajectory import (
    FootstepPlanKnotPoints,
    FootstepPlanSegment,
    FootstepTrajectory,
)
from planning_through_contact.planning.footstep.in_plane_terrain import InPlaneTerrain

GcsVertex = GraphOfConvexSets.Vertex
GcsEdge = GraphOfConvexSets.Edge


class VertexSegmentPair(NamedTuple):
    v: GcsVertex
    s: FootstepPlanSegment

    def get_var_in_vertex(self, var: Variable) -> Variable:
        return self.s.get_var_in_vertex(var, self.v.x())

    def get_vars_in_vertex(self, vars: npt.NDArray) -> npt.NDArray:
        return self.s.get_vars_in_vertex(vars, self.v.x())

    def get_vars_from_vertex_vars(self, vertex_vars: npt.NDArray) -> npt.NDArray:
        temp = MathematicalProgram()
        temp.AddDecisionVariables(self.v.x())
        idxs = temp.FindDecisionVariableIndices(vertex_vars)
        return self.s.prog.decision_variables()[
            idxs
        ]  # NOTE: This will intentionally only work for degree 1 monomial variables

    def get_lin_exprs_in_vertex(self, vars: npt.NDArray) -> npt.NDArray:
        return self.s.get_lin_exprs_in_vertex(vars, self.v.x())

    def get_knot_point_vals(
        self, result: MathematicalProgramResult
    ) -> FootstepPlanKnotPoints:
        return self.s.evaluate_with_vertex_result(result, self.v.x())

    def add_cost_to_vertex(self) -> None:
        for binding in self.s.prog.GetAllCosts():
            vertex_vars = self.s.get_vars_in_vertex(binding.variables(), self.v.x())
            new_binding = Binding[type(binding.evaluator())](
                binding.evaluator(), vertex_vars
            )
            self.v.AddCost(new_binding)


class FootstepPlanRounder:
    def __init__(
        self,
        active_edges: List[GcsEdge],
        vertex_segment_pairs: Dict[str, VertexSegmentPair],
        result: MathematicalProgram,
    ) -> None:
        """
        This object takes the result from a GCS plan and rounds it to obtain a feasible solution.
        It takes in a list of active edges, all pairs of VertexSegmentPairs in the graph, and the result
        from solving the GCS problem.

        NOTE: This class is not really specific for footstep planning and can in principle (and will most likely)
        be extended to handle any nonconvex gcs trajopt for QCQPs.
        """
        # we disregard source and target vertices when we extract the path
        active_pairs = [vertex_segment_pairs[e.v().name()] for e in active_edges[:-1]]
        self.active_edges = active_edges

        self.all_pairs = vertex_segment_pairs

        active_vertices, active_segments = zip(*active_pairs)
        self.active_segments = active_segments
        self.active_vertices = active_vertices

        self.active_pairs = [self.all_pairs[v.name()] for v in active_vertices]

        for v in active_vertices:
            if len(v.GetConstraints()) > 0:
                raise NotImplementedError(
                    "We do not currently support vertex constraints"
                )

        # Assemble one big nonlinear program from the small nonlinear programs
        self.prog = MathematicalProgram()
        for s in active_segments:
            vars = s.prog.decision_variables()
            self.prog.AddDecisionVariables(vars)

            for c in s.prog.GetAllConstraints():
                self.prog.AddConstraint(c.evaluator(), c.variables())

            for c in s.prog.GetAllCosts():
                self.prog.AddCost(c.evaluator(), c.variables())

        # Add all the edge constraints from the graph
        # (which may couple the variables from individual nonlinear programs)
        for e in active_edges:
            u, v = e.u(), e.v()

            for binding in e.GetConstraints():
                # Find the corresponding original variables given the edge constraint (which is in the set variables)
                new_vars = self._from_vertex_vars_to_prog_vars(
                    binding.variables(), u, v
                )
                exprs = binding.evaluator().Eval(binding.variables())
                new_exprs = []
                for e in exprs:
                    new_expr = e.Substitute(
                        {
                            old_var: new_var
                            for old_var, new_var in zip(binding.variables(), new_vars)
                        }
                    )
                    new_exprs.append(new_expr)

                new_exprs = np.array(new_exprs)

                lb = binding.evaluator().lower_bound()
                ub = binding.evaluator().upper_bound()
                if lb == ub:  # equality constraint
                    self.prog.AddConstraint(eq(new_exprs, lb))
                elif not np.isinf(lb):
                    self.prog.AddConstraint(ge(new_exprs, lb))
                elif not np.isinf(ub):
                    self.prog.AddConstraint(ge(new_exprs, ub))
                else:
                    raise RuntimeError(
                        "Trying to add a constraint without an upper or lower bound"
                    )

        # Set the initial guess
        self.initial_guess = self._get_initial_guess_as_orig_variables(result)

    def _from_vertex_vars_to_prog_vars(
        self, vertex_vars: npt.NDArray, u: GcsVertex, v: GcsVertex
    ) -> npt.NDArray:
        u_vars = []
        v_vars = []
        u_vars_idxs = []
        v_vars_idxs = []

        all_u_vars = Variables(u.x())
        all_v_vars = Variables(v.x())

        for idx, var in enumerate(vertex_vars):
            if var in all_u_vars:
                u_vars.append(var)
                u_vars_idxs.append(idx)
            elif var in all_v_vars:
                v_vars.append(var)
                v_vars_idxs.append(idx)
            else:
                raise RuntimeError(f"Variable {var} not in any of the vertices")

        def _find_idx_in_variables(var, vars) -> int:
            for idx, var_target in enumerate(vars):
                if var.EqualTo(var_target):
                    return idx

            raise RuntimeError("Could not find variable")

        u_idxs = [_find_idx_in_variables(var, all_u_vars) for var in u_vars]
        v_idxs = [_find_idx_in_variables(var, all_v_vars) for var in v_vars]

        # We need extra logic here to deal with the fact that initial and target constraints
        # are added as singleton sets (Point class).
        if u.name() == "source":
            u_prog_vars = u.set().x()[u_idxs]
        else:
            segment_u = self.all_pairs[u.name()].s
            u_prog_vars = segment_u.prog.decision_variables()[u_idxs]

        if v.name() == "target":
            v_prog_vars = v.set().x()[v_idxs]
        else:
            segment_v = self.all_pairs[v.name()].s
            v_prog_vars = segment_v.prog.decision_variables()[v_idxs]

        # now assemble back the variables in the right order
        all_vars_unsorted = [
            (var, idx)
            for var, idx in zip(
                np.concatenate([u_prog_vars, v_prog_vars]), u_vars_idxs + v_vars_idxs
            )
        ]
        sorted_vars = [var for var, _ in sorted(all_vars_unsorted, key=lambda p: p[1])]
        return np.array(sorted_vars)

    def _get_initial_guess_as_orig_variables(
        self,
        gcs_result: MathematicalProgramResult,
    ) -> npt.NDArray[np.float64]:
        vertex_vars = [
            p.get_vars_in_vertex(p.s.prog.decision_variables())
            for p in self.active_pairs
        ]
        all_vertex_vars_concatenated = np.concatenate(vertex_vars)
        vertex_var_vals = gcs_result.GetSolution(all_vertex_vars_concatenated)

        if not len(vertex_var_vals) == len(self.prog.decision_variables()):
            breakpoint()
            raise RuntimeError(
                "Number of vertex variables must match number of decision variables when picking initial guess"
            )
        return vertex_var_vals

    def round(self) -> MathematicalProgramResult:
        snopt = SnoptSolver()
        rounded_result = snopt.Solve(self.prog, initial_guess=self.initial_guess)  # type: ignore
        return rounded_result

    def get_plan(self, result: MathematicalProgramResult) -> FootstepTrajectory:
        knot_points = [s.evaluate_with_result(result) for s in self.active_segments]
        dt = self.active_segments[0].dt
        plan = FootstepTrajectory.from_segments(knot_points, dt)
        return plan


class FootstepPlanner:
    def __init__(
        self,
        config: FootstepPlanningConfig,
        terrain: InPlaneTerrain,
        initial_pose: npt.NDArray[np.float64],
        target_pose: npt.NDArray[np.float64],
        initial_stone_name: str = "initial",
        target_stone_name: str = "target",
    ) -> None:
        self.config = config
        self.stones = terrain.stepping_stones
        self.robot = config.robot

        self.segments_per_stone = self._make_segments_for_terrain()

        self.gcs = GraphOfConvexSets()

        # Add initial and target vertices
        self.source = self.gcs.AddVertex(Point(initial_pose), name="source")
        self.target = self.gcs.AddVertex(Point(target_pose), name="target")

        # Add all segments as vertices
        self.segment_vertex_pairs_per_stone = {}
        for stone, segments_for_stone in zip(self.stones, self.segments_per_stone):
            self.segment_vertex_pairs_per_stone[stone.name] = (
                self._add_segments_as_vertices(
                    self.gcs, segments_for_stone, self.config.use_lp_approx
                )
            )

        self.transition_pairs = self._make_transition_segments()

        # Collect all pairs in a flat dictionary that maps from segment name to segment
        self.all_segment_vertex_pairs = {
            pair.s.name: pair
            for pairs_for_stone in self.segment_vertex_pairs_per_stone.values()
            for pair in pairs_for_stone.values()
        }
        for transition_pair in self.transition_pairs.values():
            self.all_segment_vertex_pairs[transition_pair.s.name] = transition_pair

        edges_to_add = self._collect_all_graph_edges()

        self._add_edges_with_dynamics_constraints(
            self.gcs,
            edges_to_add,
            self.all_segment_vertex_pairs,
            self.config.dt,
        )
        # Connect the first segment in the initial stone to the source
        self._add_edge_to_source_or_target(
            list(self.segment_vertex_pairs_per_stone[initial_stone_name].values())[0],
            "source",
        )

        # Connect all segments with two feet on the ground on the source stone to target
        for transition_pair in self.segment_vertex_pairs_per_stone[
            target_stone_name
        ].values():
            if transition_pair.s.two_feet:
                self._add_edge_to_source_or_target(transition_pair, "target")

    @staticmethod
    def _calc_num_steps_required_per_stone(width: float, step_span: float) -> int:
        return int(np.floor(width / step_span) + 2)

    def _make_segments_for_terrain(self) -> List[List[FootstepPlanSegment]]:
        num_steps_required_per_stone = [
            self._calc_num_steps_required_per_stone(stone.width, self.robot.step_span)
            for stone in self.stones
        ]
        segments = []
        for stone, num_steps_required in zip(self.stones, num_steps_required_per_stone):
            segments_for_stone = []
            # This makes sure that we add num_steps_required steps,
            # where we start with a lift step, and end with a lift step
            for gait_idx in range(num_steps_required * 2 - 1):
                # This makes the first segment start with one foot, then alternate
                # from there
                step_idx = int(gait_idx / 2)
                one_foot_segment = (gait_idx + 1) % 2 == 1
                if one_foot_segment:
                    lift_step = FootstepPlanSegment(
                        stone,
                        "one_foot",
                        self.robot,
                        self.config,
                        name=f"step_{step_idx}_one_foot",
                    )
                    segments_for_stone.append(lift_step)
                else:
                    stance_step = FootstepPlanSegment(
                        stone,
                        "two_feet",
                        self.robot,
                        self.config,
                        name=f"step_{step_idx}_two_feet",
                    )
                    segments_for_stone.append(stance_step)

            segments.append(segments_for_stone)

        # Append one stance segment to the start of the first segment on the first stone
        stance_step = FootstepPlanSegment(
            self.stones[0],
            "two_feet",
            self.robot,
            self.config,
            name=f"start_stance",
        )
        segments[0].insert(0, stance_step)

        # Append one stance segment to the end of the last segment on the last stone
        stance_step = FootstepPlanSegment(
            self.stones[-1],
            "two_feet",
            self.robot,
            self.config,
            name=f"final_stance",
        )
        segments[-1].append(stance_step)

        return segments

    def _add_segments_as_vertices(
        self,
        gcs: GraphOfConvexSets,
        segments: List[FootstepPlanSegment],
        use_lp_approx: bool = False,
    ) -> Dict[str, VertexSegmentPair]:
        vertices = [
            gcs.AddVertex(s.get_convex_set(use_lp_approx=use_lp_approx), name=s.name)
            for s in segments
        ]
        pairs = {s.name: VertexSegmentPair(v, s) for v, s in zip(vertices, segments)}
        for pair in pairs.values():
            pair.add_cost_to_vertex()

        # Make sure there are no naming duplicates
        if not len(set(pairs.keys())) == len(pairs):
            raise RuntimeError("Names cannot be the same.")

        return pairs

    def _make_transition_segments(self) -> Dict[Tuple[str, str], VertexSegmentPair]:
        # Edges and transitions between stones
        transition_segments = []
        for stone_u, stone_v in zip(self.stones[:-1], self.stones[1:]):
            transition_step = FootstepPlanSegment(
                stone_u,
                "two_feet",
                self.robot,
                self.config,
                name=f"transition",
                stone_for_last_foot=stone_v,
            )
            transition_segments.append(transition_step)

        transition_vertices = [
            self.gcs.AddVertex(s.get_convex_set(use_lp_approx=False), name=s.name)
            for s in transition_segments
        ]
        transition_pairs = {
            (s.stone_first.name, s.stone_last.name): VertexSegmentPair(v, s)
            for v, s in zip(transition_vertices, transition_segments)
        }
        return transition_pairs

    def _collect_all_graph_edges(self) -> List[Tuple[str, str]]:
        # Create a list of all edges we should add
        forward_edges = []
        # Edges between segments within a stone
        for segments_for_stone in self.segments_per_stone:
            names = [segment.name for segment in segments_for_stone]
            edges = [(name_i, name_j) for name_i, name_j in zip(names[:-1], names[1:])]
            forward_edges.extend(edges)

        transition_edges = []
        for (
            stone_u_name,
            stone_v_name,
        ), transition_pair in self.transition_pairs.items():
            # connect all the incoming segments with only one foot in contact to the transition segment
            incoming_edges = [
                (incoming_pair.s.name, transition_pair.s.name)
                for incoming_pair in self.segment_vertex_pairs_per_stone[
                    stone_u_name
                ].values()
                if not incoming_pair.s.two_feet
            ]
            transition_edges.extend(incoming_edges)

            # connect the transition segment to the first segment of the next stone
            outgoing_edge = (
                transition_pair.s.name,
                list(self.segment_vertex_pairs_per_stone[stone_v_name].values())[
                    0
                ].s.name,
            )
            transition_edges.append(outgoing_edge)

        edges_to_add = forward_edges + transition_edges
        return edges_to_add

    def _add_edges_between_stones(
        self,
        gcs: GraphOfConvexSets,
        edges_to_add: List[Tuple[str, str]],
        pairs: Dict[str, VertexSegmentPair],
        dt: float,
    ) -> None:
        # edge from i -> j
        for i, j in edges_to_add:
            pair_u, pair_v = pairs[i], pairs[j]
            u, s_u = pair_u
            v, s_v = pair_v

            e = gcs.AddEdge(u, v)

            state_curr = s_u.get_vars_in_vertex(s_u.get_state(-1), u.x())
            f_curr = s_u.get_lin_exprs_in_vertex(s_u.get_dynamics(-1), u.x())
            state_next = s_v.get_vars_in_vertex(s_v.get_state(0), v.x())

            # forward euler
            constraint = eq(state_next, state_curr + dt * f_curr)
            for c in constraint:
                e.AddConstraint(c)

    def _add_edges_with_dynamics_constraints(
        self,
        gcs: GraphOfConvexSets,
        edges_to_add: List[Tuple[str, str]],
        pairs: Dict[str, VertexSegmentPair],
        dt: float,
    ) -> None:
        # edge from i -> j
        for i, j in edges_to_add:
            pair_u, pair_v = pairs[i], pairs[j]
            u, s_u = pair_u
            v, s_v = pair_v

            e = gcs.AddEdge(u, v)

            state_curr = s_u.get_vars_in_vertex(s_u.get_state(-1), u.x())
            f_curr = s_u.get_lin_exprs_in_vertex(s_u.get_dynamics(-1), u.x())
            state_next = s_v.get_vars_in_vertex(s_v.get_state(0), v.x())

            # forward euler
            constraint = eq(state_next, state_curr + dt * f_curr)
            for c in constraint:
                e.AddConstraint(c)

            if (s_u.two_feet and s_v.two_feet) or (
                not s_u.two_feet and not s_v.two_feet
            ):
                raise RuntimeError(
                    "Must transition from stance -> lift or lift -> stance between two segments"
                )

            # If the segment we are transitining from has both feet in contact,
            # we constrain the last foot to be constant. Otherwise, we are
            # transitioning to a mode with both feet, and constrain the first
            # to be constant
            if s_u.two_feet:
                constant_foot = "last"
            else:
                constant_foot = "first"

            # Foot in contact cant move
            constant_foot_u_x_pos = pair_u.get_var_in_vertex(
                s_u.get_foot_pos(constant_foot, -1)
            )
            constant_foot_v_x_pos = pair_v.get_var_in_vertex(
                s_v.get_foot_pos(constant_foot, 0)
            )
            e.AddConstraint(constant_foot_u_x_pos == constant_foot_v_x_pos)  # type: ignore

            # TODO: Need to also add cost on edge (right now the dynamics step between two contacts is "free")

    def _add_edge_to_source_or_target(
        self,
        pair: VertexSegmentPair,
        source_or_target: Literal["source", "target"] = "source",
    ) -> None:
        if source_or_target == "source":
            s = self.source
            # source -> v
            e = self.gcs.AddEdge(s, pair.v)
            pose = pair.get_vars_in_vertex(pair.s.get_robot_pose(0))
            spatial_vel = pair.get_vars_in_vertex(pair.s.get_robot_spatial_vel(0))
            spatial_acc = pair.get_lin_exprs_in_vertex(pair.s.get_robot_spatial_acc(0))
        else:  # target
            s = self.target
            # v -> target
            e = self.gcs.AddEdge(pair.v, s)
            pose = pair.get_vars_in_vertex(pair.s.get_robot_pose(-1))
            spatial_vel = pair.get_vars_in_vertex(pair.s.get_robot_spatial_vel(-1))
            spatial_acc = pair.get_lin_exprs_in_vertex(pair.s.get_robot_spatial_acc(-1))

        # The only variables in the source/target are the pose variables
        constraint = eq(pose, s.x())
        for c in constraint:
            e.AddConstraint(c)

        # TODO: I don't think that this makes much sense to have
        # Add zero velocity constraint on the edge connection connected to the source or target
        # constraint = eq(spatial_vel, 0)
        # for c in constraint:
        #     e.AddConstraint(c)

        # Enforce that we start and end in an equilibrium position
        constraint = eq(spatial_acc, 0)
        for c in constraint:
            e.AddConstraint(c)

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

    def plan(
        self, print_flows: bool = False, print_solver_output: bool = False
    ) -> FootstepTrajectory:
        options = GraphOfConvexSetsOptions()
        options.convex_relaxation = True
        options.max_rounded_paths = 20

        solver_options = SolverOptions()
        if print_solver_output:
            solver_options.SetOption(CommonSolverOption.kPrintToConsole, 1)  # type: ignore
            options.solver_options = solver_options

        # tolerance = 1e-6
        # mosek = MosekSolver()
        # options.solver = mosek
        # solver_options.SetOption(
        #     mosek.solver_id(), "MSK_DPAR_INTPNT_CO_TOL_PFEAS", tolerance
        # )
        # solver_options.SetOption(
        #     mosek.solver_id(), "MSK_DPAR_INTPNT_CO_TOL_DFEAS", tolerance
        # )
        # solver_options.SetOption(
        #     mosek.solver_id(), "MSK_DPAR_INTPNT_CO_TOL_REL_GAP", tolerance
        # )

        print("Solving GCS problem")
        gcs_result = self.gcs.SolveShortestPath(self.source, self.target, options)

        if not gcs_result.is_success():
            print("GCS problem failed to solve")
            # TODO
        # TODO remove this
        gcs_result.set_solution_result(SolutionResult.kSolutionFound)

        if print_flows:
            flows = [
                (e.name(), gcs_result.GetSolution(e.phi())) for e in self.gcs.Edges()
            ]
            flow_strings = [f"{name}: {val:.2f}" for name, val in flows]
            print(f"Graph flows: {', '.join(flow_strings)}")

        paths, relaxed_results = self.gcs.GetRandomizedSolutionPath(
            self.source, self.target, gcs_result, options
        )
        rounders = []
        rounded_results = []

        MAX_ROUNDINGS = 3

        print(f"Rounding {len(paths)} possible GCS paths...")
        for idx, (active_edges, relaxed_result) in enumerate(
            zip(paths, relaxed_results)
        ):
            print(f"Rounding_step: {idx}")
            if idx >= MAX_ROUNDINGS:
                break
            rounder = FootstepPlanRounder(
                active_edges, self.all_segment_vertex_pairs, relaxed_result
            )
            rounded_result = rounder.round()
            rounded_results.append(rounded_result)
            rounders.append(rounder)

        best_idx = np.argmin(
            [
                res.get_optimal_cost() if res.is_success() else np.inf
                for res in rounded_results
            ]
        )
        rounder, rounded_result = rounders[best_idx], rounded_results[best_idx]
        assert rounded_result.is_success()

        active_edge_names = [e.name() for e in rounder.active_edges]
        print(f"Best path: {' -> '.join(active_edge_names)}")

        c_round = rounded_result.get_optimal_cost()
        c_relax = gcs_result.get_optimal_cost()
        ub_optimality_gap = (c_round - c_relax) / c_relax
        print(f"UB optimality gap: {ub_optimality_gap:.5f} %")

        plan = rounder.get_plan(rounded_result)

        return plan
