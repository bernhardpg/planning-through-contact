from typing import Dict, List, Literal, NamedTuple, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pydot
from pydrake.geometry.optimization import (
    GraphOfConvexSets,
    GraphOfConvexSetsOptions,
    Point,
)
from pydrake.math import eq
from pydrake.solvers import (
    Binding,
    CommonSolverOption,
    MathematicalProgram,
    MathematicalProgramResult,
    SolutionResult,
    SolverOptions,
)

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
        """
        # we disregard source and target vertices when we extract the path
        active_pairs = [vertex_segment_pairs[e.v().name()] for e in active_edges[:-1]]

        vertices, segments = zip(*active_pairs)
        self.segments = segments
        self.vertices = vertices

        # Assemble one big nonlinear program from the small nonlinear programs
        self.prog = MathematicalProgram()
        for s in segments:
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

            # TODO: continue here

            if u.name() == "source":
                breakpoint()

            breakpoint()

        solution_gait_schedule = np.vstack([p.s.active_feet for p in active_pairs])


class FootstepPlanner:
    def __init__(
        self,
        config: FootstepPlanningConfig,
        terrain: InPlaneTerrain,
        initial_pose: npt.NDArray[np.float64],
        target_pose: npt.NDArray[np.float64],
    ) -> None:
        self.config = config

        initial_stone = terrain.stepping_stones[0]
        # target_stone = terrain.stepping_stones[1]

        robot = config.robot
        dt = config.dt

        gait_schedule = np.array([[1, 1]])
        segments = [
            FootstepPlanSegment(
                initial_stone,
                foot_activation,
                robot,
                config,
                name=str(idx),
            )
            for idx, foot_activation in enumerate(gait_schedule)
        ]

        self.gait_schedule = gait_schedule

        self.gcs = GraphOfConvexSets()

        # Add initial and target vertices
        self.source = self.gcs.AddVertex(Point(initial_pose), name="source")
        self.target = self.gcs.AddVertex(Point(target_pose), name="target")

        # Add all knot points as vertices
        pairs = self._add_segments_as_vertices(self.gcs, segments)

        # edges_to_add = []
        # self._add_edges_with_dynamics_constraints(self.gcs, edges_to_add, pairs, dt)

        # TODO: Continuity constraints on subsequent contacts within the same region

        self._add_edge_to_source_or_target(pairs[0], "source")

        for pair in pairs:  # connect all the vertices to the target
            self._add_edge_to_source_or_target(pair, "target")

        self.vertex_name_to_pairs = {pair.v.name(): pair for pair in pairs}

    def _add_segments_as_vertices(
        self, gcs: GraphOfConvexSets, segments: List[FootstepPlanSegment]
    ) -> List[VertexSegmentPair]:
        vertices = [gcs.AddVertex(s.get_convex_set(), name=s.name) for s in segments]
        pairs = [VertexSegmentPair(v, s) for v, s in zip(vertices, segments)]
        for pair in pairs:
            pair.add_cost_to_vertex()

        return pairs

    def _add_edges_with_dynamics_constraints(
        self,
        gcs: GraphOfConvexSets,
        edges_to_add: List[Tuple[int, int]],
        pairs: List[VertexSegmentPair],
        dt: float,
    ) -> None:
        # edge from i -> j
        for i, j in edges_to_add:
            u, s_u = pairs[i]
            v, s_v = pairs[j]

            e = gcs.AddEdge(u, v)

            state_curr = s_u.get_vars_in_vertex(s_u.get_state(-1), u.x())
            f_curr = s_u.get_lin_exprs_in_vertex(s_u.get_dynamics(-1), u.x())
            state_next = s_v.get_vars_in_vertex(s_v.get_state(0), v.x())

            # forward euler
            # constraint = eq(state_next, state_curr + dt * f_curr)
            # for c in constraint:
            #     e.AddConstraint(c)

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

    def plan(self) -> FootstepTrajectory:
        options = GraphOfConvexSetsOptions()
        options.convex_relaxation = True
        options.max_rounded_paths = 20

        solver_options = SolverOptions()
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

        result = self.gcs.SolveShortestPath(self.source, self.target, options)

        if not result.is_success():
            # raise RuntimeError("Could not find a solution!")
            print("Could not find a feasible solution!")

        # TODO remove this
        result.set_solution_result(SolutionResult.kSolutionFound)

        flows = {e.name(): result.GetSolution(e.phi()) for e in self.gcs.Edges()}
        print(flows)

        if False:
            paths, results = self.gcs.GetRandomizedSolutionPath(
                self.source, self.target, result, options
            )
            active_edges = paths[0]
            result = results[0]

        active_edges = self.gcs.GetSolutionPath(self.source, self.target, result)
        active_edge_names = [e.name() for e in active_edges]
        print(f"Path: {' -> '.join(active_edge_names)}")

        rounder = FootstepPlanRounder(active_edges, self.vertex_name_to_pairs, result)

        # we disregard source and target vertices when we extract the path
        pairs_on_sol = [
            self.vertex_name_to_pairs[e.v().name()] for e in active_edges[:-1]
        ]

        solution_gait_schedule = np.vstack([p.s.active_feet for p in pairs_on_sol])

        segments = [p.get_knot_point_vals(result, round=True) for p in pairs_on_sol]
        breakpoint()
        plan = FootstepTrajectory.from_segments(
            segments, self.config.dt, solution_gait_schedule
        )
        print(f"Cost: {result.get_optimal_cost()}")

        return plan
