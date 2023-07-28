from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pydrake.geometry.optimization as opt
import pydrake.symbolic as sym
from pydrake.math import eq, ge
from pydrake.solvers import (
    Binding,
    BoundingBoxConstraint,
    Cost,
    LinearConstraint,
    LinearCost,
    MathematicalProgram,
    MathematicalProgramResult,
    QuadraticCost,
    Solve,
)

from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    PolytopeContactLocation,
)
from planning_through_contact.geometry.planar.abstract_mode import (
    AbstractContactMode,
    AbstractModeVariables,
    ContinuityVariables,
)
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.rigid_body import RigidBody
from planning_through_contact.planning.planar.planar_plan_specs import PlanarPlanSpecs
from planning_through_contact.tools.types import NpVariableArray

GcsVertex = opt.GraphOfConvexSets.Vertex
GcsEdge = opt.GraphOfConvexSets.Edge


def check_pos_in_contact_location(
    pos: npt.NDArray[np.float64],
    loc: PolytopeContactLocation,
    body: RigidBody,
    body_pose: PlanarPose,
) -> bool:
    specs = PlanarPlanSpecs()
    mode = NonCollisionMode.create_from_plan_spec(loc, specs, body, one_knot_point=True)

    mode.set_finger_initial_pos(pos)
    mode.set_slider_pose(body_pose)

    result = Solve(mode.prog)
    return result.is_success()


@dataclass
class NonCollisionVariables(AbstractModeVariables):
    p_BF_xs: NpVariableArray | npt.NDArray[np.float64]
    p_BF_ys: NpVariableArray | npt.NDArray[np.float64]
    p_WB_x: sym.Variable | float
    p_WB_y: sym.Variable | float
    cos_th: sym.Variable | float
    sin_th: sym.Variable | float

    @classmethod
    def from_prog(
        cls, prog: MathematicalProgram, num_knot_points: int, time_in_mode: float
    ) -> "NonCollisionVariables":
        # Finger location
        p_BF_xs = prog.NewContinuousVariables(num_knot_points, "p_BF_x")
        p_BF_ys = prog.NewContinuousVariables(num_knot_points, "p_BF_y")

        # We only need one variable for the pose of the object
        p_WB_x = prog.NewContinuousVariables(1, "p_WB_x").item()
        p_WB_y = prog.NewContinuousVariables(1, "p_WB_y").item()
        cos_th = prog.NewContinuousVariables(1, "cos_th").item()
        sin_th = prog.NewContinuousVariables(1, "sin_th").item()

        dt = time_in_mode / num_knot_points

        return NonCollisionVariables(
            num_knot_points,
            time_in_mode,
            dt,
            p_BF_xs,
            p_BF_ys,
            p_WB_x,
            p_WB_y,
            cos_th,
            sin_th,
        )

    def eval_result(self, result: MathematicalProgramResult) -> "NonCollisionVariables":
        return NonCollisionVariables(
            self.num_knot_points,
            self.time_in_mode,
            self.dt,
            result.GetSolution(self.p_BF_xs),
            result.GetSolution(self.p_BF_ys),
            result.GetSolution(self.p_WB_x),  # type: ignore
            result.GetSolution(self.p_WB_y),  # type: ignore
            result.GetSolution(self.cos_th),  # type: ignore
            result.GetSolution(self.sin_th),  # type: ignore
        )

    @property
    def p_BFs(self):
        return [
            np.expand_dims(np.array([x, y]), 1)
            for x, y in zip(self.p_BF_xs, self.p_BF_ys)
        ]  # (2, 1)

    @property
    def p_WB(self):
        return np.expand_dims(np.array([self.p_WB_x, self.p_WB_y]), 1)  # (2, 1)

    @property
    def R_WBs(self):
        Rs = [
            np.array([[self.cos_th, -self.sin_th], [self.sin_th, self.cos_th]])
        ] * self.num_knot_points
        return Rs

    @property
    def p_WBs(self):
        return [self.p_WB] * self.num_knot_points

    @property
    def v_WBs(self):
        NUM_DIMS = 2
        return [np.zeros((NUM_DIMS, 1))] * (self.num_knot_points - 1)

    @property
    def omega_WBs(self):
        return [0.0] * (self.num_knot_points - 1)

    @property
    def p_c_Ws(self):
        return [
            p_WB + R_WB.dot(p_c_B)
            for p_WB, R_WB, p_c_B in zip(self.p_WBs, self.R_WBs, self.p_BFs)
        ]

    @property
    def f_c_Ws(self):
        NUM_DIMS = 2
        return [np.zeros((NUM_DIMS, 1))] * self.num_knot_points

    @property
    def p_c_Bs(self):
        return self.p_BFs


@dataclass
class NonCollisionMode(AbstractContactMode):
    avoid_object: bool = False
    avoidance_cost: Literal["linear", "quadratic"] = "quadratic"
    avoidance_param_linear: float = 0.1
    avoidance_param_quadratic: float = 0.2

    @classmethod
    def create_from_plan_spec(
        cls,
        contact_location: PolytopeContactLocation,
        specs: PlanarPlanSpecs,
        slider: RigidBody,
        name: Optional[str] = None,
        one_knot_point: bool = False,
        avoid_object: bool = False,
    ) -> "NonCollisionMode":
        if name is None:
            name = f"NON_COLL_{contact_location.idx}"

        num_knot_points = 1 if one_knot_point else specs.num_knot_points_non_collision

        return cls(
            name,
            num_knot_points,
            specs.time_non_collision,
            contact_location,
            slider,
            avoid_object,
        )

    def __post_init__(self) -> None:
        self.dt = self.time_in_mode / self.num_knot_points

        self.planes = self.object.geometry.get_planes_for_collision_free_region(
            self.contact_location
        )
        self.prog = MathematicalProgram()
        self.variables = NonCollisionVariables.from_prog(
            self.prog, self.num_knot_points, self.time_in_mode
        )
        self._define_constraints()
        self._define_cost()

    def _define_constraints(self) -> None:
        for k in range(self.num_knot_points):
            p_BF = self.variables.p_BFs[k]

            for plane in self.planes:
                self.prog.AddLinearConstraint(plane.dist_to(p_BF) >= 0)

    def _define_cost(self) -> None:
        position_diffs = [
            p_next - p_curr
            for p_next, p_curr in zip(
                self.variables.p_BFs[1:], self.variables.p_BFs[:-1]
            )
        ]
        if self.num_knot_points == 1:
            # Can't vstack only one array
            position_diffs = np.array(position_diffs)
        else:
            position_diffs = np.vstack(position_diffs)
        # position_diffs is now one long vector with diffs in each entry
        squared_eucl_dist = position_diffs.T.dot(position_diffs).item()
        self.prog.AddQuadraticCost(squared_eucl_dist, is_convex=True)

        if self.avoid_object:
            plane = self.object.geometry.faces[self.contact_location.idx]
            dists = [plane.dist_to(p_BF) for p_BF in self.variables.p_BFs]
            if self.avoidance_cost == "linear":
                self.prog.AddLinearCost(
                    -self.avoidance_param_linear * np.sum(dists)
                )  # maximize distances
            else:  # quadratic
                squared_dists = [
                    (d - self.avoidance_param_quadratic) ** 2 for d in dists
                ]
                self.prog.AddQuadraticCost(np.sum(squared_dists), is_convex=True)

    def set_slider_pose(self, pose: PlanarPose) -> None:
        self.slider_pose = pose
        self.prog.AddLinearConstraint(self.variables.cos_th == np.cos(pose.theta))
        self.prog.AddLinearConstraint(self.variables.sin_th == np.sin(pose.theta))
        self.prog.AddLinearConstraint(eq(self.variables.p_WB, pose.pos()))

    def set_finger_initial_pos(self, pos: npt.NDArray[np.float64]) -> None:
        self.p_BF_initial = pos
        self.prog.AddLinearConstraint(eq(self.variables.p_BFs[0], pos))

    def set_finger_final_pos(self, pos: npt.NDArray[np.float64]) -> None:
        self.p_BF_final = pos
        self.prog.AddLinearConstraint(eq(self.variables.p_BFs[-1], pos))

    def get_variable_indices_in_gcs_vertex(self, vars: NpVariableArray) -> List[int]:
        return self.prog.FindDecisionVariableIndices(vars)

    def get_variable_solutions_for_vertex(
        self, vertex: GcsVertex, result: MathematicalProgramResult
    ) -> NonCollisionVariables:
        # TODO: This can probably be cleaned up somehow
        p_BF_xs = self._get_vars_solution_for_vertex_vars(vertex.x(), self.variables.p_BF_xs, result)  # type: ignore
        p_BF_ys = self._get_vars_solution_for_vertex_vars(vertex.x(), self.variables.p_BF_ys, result)  # type: ignore
        p_WB_x = self._get_var_solution_for_vertex_vars(vertex.x(), self.variables.p_WB_x, result)  # type: ignore
        p_WB_y = self._get_var_solution_for_vertex_vars(vertex.x(), self.variables.p_WB_y, result)  # type: ignore
        cos_th = self._get_var_solution_for_vertex_vars(vertex.x(), self.variables.cos_th, result)  # type: ignore
        sin_th = self._get_var_solution_for_vertex_vars(vertex.x(), self.variables.sin_th, result)  # type: ignore
        return NonCollisionVariables(
            self.variables.num_knot_points,
            self.variables.time_in_mode,
            self.variables.dt,
            p_BF_xs,
            p_BF_ys,
            p_WB_x,
            p_WB_y,
            cos_th,
            sin_th,
        )

    def get_convex_set(self, make_bounded: bool = True) -> opt.Spectrahedron:
        # Create a temp program without a quadratic cost that we can use to create a polyhedron
        temp_prog = MathematicalProgram()
        x = temp_prog.NewContinuousVariables(self.prog.num_vars(), "x")
        # Some linear constraints will be added as bounding box constraints
        for c in self.prog.GetAllConstraints():
            if not (
                isinstance(c.evaluator(), LinearConstraint)
                or isinstance(c.evaluator(), BoundingBoxConstraint)
            ):
                raise ValueError("Constraints must be linear!")

            idxs = self.get_variable_indices_in_gcs_vertex(c.variables())
            vars = x[idxs]
            temp_prog.AddConstraint(c.evaluator(), vars)

        # GCS requires the sets to be bounded
        if make_bounded:
            BOUND = 1  # TODO(bernhardpg): this should not be hardcoded
            ub = np.full((temp_prog.num_vars(),), BOUND)
            temp_prog.AddBoundingBoxConstraint(-ub, ub, temp_prog.decision_variables())

        # NOTE: Here, we are using the Spectrahedron constructor, which is really creating a polyhedron,
        # because there is no PSD constraint. In the future, it is cleaner to use an interface for the HPolyhedron class.
        poly = opt.Spectrahedron(temp_prog)

        # NOTE: They sets will likely be unbounded

        return poly

    def get_convex_set_in_positions(self) -> opt.Spectrahedron:
        # Construct a small temporary program in R^2 that will allow us to check
        # for positional intersections between regions
        NUM_DIMS = 2
        temp_prog = MathematicalProgram()
        x = temp_prog.NewContinuousVariables(NUM_DIMS, "x")

        for plane in self.planes:
            dist_to_face = plane.a.T.dot(x) - plane.b  # a'x >= b
            temp_prog.AddLinearConstraint(ge(dist_to_face, 0))

        # NOTE: Here, we are using the Spectrahedron constructor, which is really creating a polyhedron,
        # because there is no PSD constraint. In the future, it is cleaner to use an interface for the HPolyhedron class.
        # TODO: Replace this with an interface to the HPolyhedron class, once this is implemented in Drake.
        poly = opt.Spectrahedron(temp_prog)

        # NOTE: They sets will likely be unbounded
        return poly

    def get_continuity_vars(
        self, first_or_last: Literal["first", "last"]
    ) -> ContinuityVariables:
        if first_or_last == "first":
            return ContinuityVariables(
                self.variables.p_BFs[0],
                self.variables.p_WB,
                self.variables.cos_th,  # type: ignore
                self.variables.sin_th,  # type: ignore
            )
        else:
            return ContinuityVariables(
                self.variables.p_BFs[-1],
                self.variables.p_WB,
                self.variables.cos_th,  # type: ignore
                self.variables.sin_th,  # type: ignore
            )

    # TODO(bernhardpg): refactor common code
    def _get_eucl_dist_cost_term(self) -> Tuple[List[int], QuadraticCost]:
        if self.avoid_object and self.avoidance_cost == "quadratic":
            assert len(self.prog.quadratic_costs()) == 2
        else:
            assert len(self.prog.quadratic_costs()) == 1

        eucl_dist_cost = self.prog.quadratic_costs()[0]  # should only be one cost
        var_idxs = self.get_variable_indices_in_gcs_vertex(eucl_dist_cost.variables())
        return var_idxs, eucl_dist_cost.evaluator()

    # TODO(bernhardpg): refactor common code
    def _get_object_avoidance_cost_term(
        self,
    ) -> Tuple[List[int], LinearCost | QuadraticCost]:
        if self.avoidance_cost == "linear":
            assert len(self.prog.linear_costs()) == 1
            object_avoidance_cost = self.prog.linear_costs()[0]
        else:  # quadratic
            assert len(self.prog.quadratic_costs()) == 2
            object_avoidance_cost = self.prog.quadratic_costs()[1]

        var_idxs = self.get_variable_indices_in_gcs_vertex(
            object_avoidance_cost.variables()
        )
        return var_idxs, object_avoidance_cost.evaluator()

    def add_cost_to_vertex(self, vertex: GcsVertex) -> None:
        # euclidean distance cost
        var_idxs, evaluator = self._get_eucl_dist_cost_term()
        vars = vertex.x()[var_idxs]
        binding = Binding[QuadraticCost](evaluator, vars)
        vertex.AddCost(binding)

        if self.avoid_object:
            var_idxs, evaluator = self._get_object_avoidance_cost_term()
            vars = vertex.x()[var_idxs]
            cost_type = LinearCost if self.avoidance_cost == "linear" else QuadraticCost
            binding = Binding[cost_type](evaluator, vars)
            vertex.AddCost(binding)
