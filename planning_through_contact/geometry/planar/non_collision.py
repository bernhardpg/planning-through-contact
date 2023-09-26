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
    LinearConstraint,
    LinearCost,
    LorentzConeConstraint,
    MathematicalProgram,
    MathematicalProgramResult,
    PerspectiveQuadraticCost,
    QuadraticCost,
    RotatedLorentzConeConstraint,
    Solve,
)

from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    ContactLocation,
    PolytopeContactLocation,
)
from planning_through_contact.geometry.planar.abstract_mode import (
    AbstractContactMode,
    AbstractModeVariables,
    ContinuityVariables,
)
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.rigid_body import RigidBody
from planning_through_contact.planning.planar.planar_plan_config import PlanarPlanConfig
from planning_through_contact.tools.types import NpVariableArray

GcsVertex = opt.GraphOfConvexSets.Vertex
GcsEdge = opt.GraphOfConvexSets.Edge


def check_finger_pose_in_contact_location(
    finger_pose: PlanarPose,
    loc: PolytopeContactLocation,
    config: PlanarPlanConfig,
) -> bool:
    mode = NonCollisionMode.create_from_plan_spec(loc, config, one_knot_point=True)

    mode.set_finger_initial_pose(finger_pose)

    result = Solve(mode.prog)
    return result.is_success()


def find_first_matching_location(
    finger_pose: PlanarPose, config: PlanarPlanConfig
) -> PolytopeContactLocation:
    # we always add all non-collision modes, even when we don't add all contact modes
    # (think of maneuvering around the object etc)
    locations = [
        PolytopeContactLocation(ContactLocation.FACE, idx)
        for idx in range(config.slider_geometry.num_collision_free_regions)
    ]
    matching_locs = [
        loc
        for loc in locations
        if check_finger_pose_in_contact_location(finger_pose, loc, config)
    ]
    if len(matching_locs) == 0:
        raise ValueError(
            "No valid configurations found for specified initial or target poses"
        )
    return matching_locs[0]


@dataclass
class NonCollisionVariables(AbstractModeVariables):
    p_BP_xs: NpVariableArray | npt.NDArray[np.float64]
    p_BP_ys: NpVariableArray | npt.NDArray[np.float64]
    p_WB_x: sym.Variable | float
    p_WB_y: sym.Variable | float
    cos_th: sym.Variable | float
    sin_th: sym.Variable | float

    @classmethod
    def from_prog(
        cls,
        prog: MathematicalProgram,
        num_knot_points: int,
        time_in_mode: float,
        contact_location: PolytopeContactLocation,
        pusher_radius: float,
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
            contact_location,
            num_knot_points,
            time_in_mode,
            dt,
            pusher_radius,
            p_BF_xs,
            p_BF_ys,
            p_WB_x,
            p_WB_y,
            cos_th,
            sin_th,
        )

    def eval_result(self, result: MathematicalProgramResult) -> "NonCollisionVariables":
        return NonCollisionVariables(
            self.contact_location,
            self.num_knot_points,
            self.time_in_mode,
            self.dt,
            self.pusher_radius,
            result.GetSolution(self.p_BP_xs),
            result.GetSolution(self.p_BP_ys),
            result.GetSolution(self.p_WB_x),  # type: ignore
            result.GetSolution(self.p_WB_y),  # type: ignore
            result.GetSolution(self.cos_th),  # type: ignore
            result.GetSolution(self.sin_th),  # type: ignore
        )

    @property
    def p_BPs(self):
        return [
            np.expand_dims(np.array([x, y]), 1)
            for x, y in zip(self.p_BP_xs, self.p_BP_ys)
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
    def p_WPs(self):
        return [
            p_WB + R_WB.dot(p_BP)
            for p_WB, R_WB, p_BP in zip(self.p_WBs, self.R_WBs, self.p_BPs)
        ]

    @property
    def f_c_Ws(self):
        NUM_DIMS = 2
        return [np.zeros((NUM_DIMS, 1))] * self.num_knot_points


@dataclass
class NonCollisionMode(AbstractContactMode):
    @classmethod
    def create_from_plan_spec(
        cls,
        contact_location: PolytopeContactLocation,
        config: PlanarPlanConfig,
        name: Optional[str] = None,
        one_knot_point: bool = False,
    ) -> "NonCollisionMode":
        if name is None:
            name = f"NON_COLL_{contact_location.idx}"

        num_knot_points = 1 if one_knot_point else config.num_knot_points_non_collision

        prog = MathematicalProgram()

        return cls(
            name,
            num_knot_points,
            config.time_non_collision,
            contact_location,
            prog,
            config,
        )

    @classmethod
    def create_source_or_target_mode(
        cls,
        config: PlanarPlanConfig,
        slider_pose: PlanarPose,
        pusher_pose: PlanarPose,
        initial_or_final: Literal["initial", "final"],
    ) -> "NonCollisionMode":
        loc = find_first_matching_location(pusher_pose, config)
        mode_name = "source" if initial_or_final == "initial" else "target"
        mode = cls.create_from_plan_spec(
            loc,
            config,
            mode_name,
            one_knot_point=True,
        )
        mode.set_slider_pose(slider_pose)

        if initial_or_final == "initial":
            mode.set_finger_initial_pose(pusher_pose)
        else:  # final
            mode.set_finger_final_pose(pusher_pose)

        return mode

    def __post_init__(self) -> None:
        self.slider_geometry = self.config.dynamics_config.slider.geometry
        self.dynamics_config = self.config.dynamics_config

        self.dt = self.time_in_mode / self.num_knot_points

        self.contact_planes = self.slider_geometry.get_contact_planes(
            self.contact_location.idx
        )
        # TODO(bernhardpg): This class should not have a contact_location object. it is not accurate,
        # as it really only has the index of a collision free set, which may or may not correspond
        # 1-1 to a
        self.collision_free_space_planes = (
            self.slider_geometry.get_planes_for_collision_free_region(
                self.contact_location.idx
            )
        )
        self.variables = NonCollisionVariables.from_prog(
            self.prog,
            self.num_knot_points,
            self.time_in_mode,
            self.contact_location,
            self.dynamics_config.pusher_radius,
        )
        self._define_constraints()
        self._define_cost()

    def _define_constraints(self) -> None:
        for k in range(self.num_knot_points):
            p_BF = self.variables.p_BPs[k]

            exprs = self._create_collision_free_space_constraints(p_BF)
            for expr in exprs:
                self.prog.AddLinearConstraint(expr)

        # TODO(bernhardpg): As of now we don't worry about the workspace constraints
        use_workspace_constraints = False
        if use_workspace_constraints:
            self._add_workspace_constraints()

    def _add_workspace_constraints(self) -> None:
        for k in range(self.num_knot_points):
            p_BF = self.variables.p_BPs[k]

            lb, ub = self.config.workspace.pusher.bounds
            self.prog.AddBoundingBoxConstraint(lb, ub, p_BF)

    def _create_collision_free_space_constraints(
        self, pusher_pos: NpVariableArray
    ) -> List[sym.Formula]:
        avoid_contact = [
            plane.dist_to(pusher_pos) - self.dynamics_config.pusher_radius >= 0
            for plane in self.contact_planes
        ]
        stay_in_region = [
            plane.dist_to(pusher_pos) >= 0 for plane in self.collision_free_space_planes
        ]
        exprs = avoid_contact + stay_in_region
        return exprs

    def _define_cost(self) -> None:
        if self.config.minimize_squared_eucl_dist:
            if self.num_knot_points > 1:
                position_diffs = [
                    p_next - p_curr
                    for p_next, p_curr in zip(
                        self.variables.p_BPs[1:], self.variables.p_BPs[:-1]
                    )
                ]
                position_diffs = np.vstack(position_diffs)
                # position_diffs is now one long vector with diffs in each entry
                squared_eucl_dist = position_diffs.T.dot(position_diffs).item()
                self.prog.AddQuadraticCost(
                    self.config.cost_terms.cost_param_eucl * squared_eucl_dist,
                    is_convex=True,
                )

        else:  # Minimize total Euclidean distance
            position_diffs = [
                p_next - p_curr
                for p_next, p_curr in zip(
                    self.variables.p_BPs[1:], self.variables.p_BPs[:-1]
                )
            ]
            slacks = self.prog.NewContinuousVariables(len(position_diffs), "t")
            for d, s in zip(position_diffs, slacks):
                # Let d := diff
                # we want to minimize the Euclidean distance:
                #   minimize sqrt(d_1^2 + d_2^2)
                # reformulate as:
                #   minimize s s.t. s >= sqrt(d_1^2 + d_2^2)
                # which is exactly a Lorentz cone constraint:
                # (s,d) \in LorentzCone <=> s >= sqrt(d_1^2 + d_2^2)
                vec = np.vstack([[s], d]).flatten()  # (s, x_diff, y_diff)
                self.prog.AddLorentzConeConstraint(vec)
                self.prog.AddLinearCost(s)

        if self.config.avoid_object:
            planes = self.slider_geometry.get_contact_planes(self.contact_location.idx)
            dists_for_each_plane = [
                [plane.dist_to(p_BF) for p_BF in self.variables.p_BPs]
                for plane in planes
            ]
            if self.config.avoidance_cost == "linear":
                raise NotImplementedError("Will be removed!")
                self.prog.AddLinearCost(
                    -self.config.cost_terms.cost_param_avoidance_lin * np.sum(dists)
                )  # maximize distances

            elif self.config.avoidance_cost == "quadratic":
                squared_dists = [
                    self.config.cost_terms.cost_param_avoidance_quad_weight
                    * (d - self.config.cost_terms.cost_param_avoidance_quad_dist) ** 2
                    for dist in dists_for_each_plane
                    for d in dist
                ]
                self.prog.AddQuadraticCost(np.sum(squared_dists), is_convex=True)

            elif self.config.avoidance_cost == "socp_single_mode":
                for dists in dists_for_each_plane:
                    dists_except_first_and_last = dists[1:-1]
                    slacks = self.prog.NewContinuousVariables(
                        len(dists_except_first_and_last), "s"
                    )
                    for d, s in zip(dists_except_first_and_last, slacks):
                        # Let d := dist >= 0
                        # we want infinite cost for being close to object:
                        #   minimize 1 / d
                        # reformulate as:
                        #   minimize s s.t. s >= 1/d, d >= 0 (implies s >= 0)
                        #   <=>
                        #   minimize s s.t. s d >= 1, d >= 0
                        # which is exactly a rotated Lorentz cone constraint:
                        # (s,d,1) \in RotatedLorentzCone <=> s d >= 1^2, s >= 0, d >= 0

                        self.prog.AddRotatedLorentzConeConstraint(np.array([s, d, 1]))
                        self.prog.AddLinearCost(
                            self.config.cost_terms.cost_param_avoidance_socp_weight * s
                        )

    def set_slider_pose(self, pose: PlanarPose) -> None:
        self.slider_pose = pose
        self.prog.AddLinearConstraint(self.variables.cos_th == np.cos(pose.theta))
        self.prog.AddLinearConstraint(self.variables.sin_th == np.sin(pose.theta))
        self.prog.AddLinearConstraint(eq(self.variables.p_WB, pose.pos()))

    def set_finger_initial_pose(self, pose: PlanarPose) -> None:
        """
        NOTE: Only sets the position of the finger (a point finger has no rotation).
        """
        self.finger_initial_pose = pose
        self.prog.AddLinearConstraint(eq(self.variables.p_BPs[0], pose.pos()))

    def set_finger_final_pose(self, pose: PlanarPose) -> None:
        """
        NOTE: Only sets the position of the finger (a point finger has no rotation).
        """
        self.finger_final_pose = pose
        self.prog.AddLinearConstraint(eq(self.variables.p_BPs[-1], pose.pos()))

    def get_variable_indices_in_gcs_vertex(self, vars: NpVariableArray) -> List[int]:
        return self.prog.FindDecisionVariableIndices(vars)

    def get_variable_solutions_for_vertex(
        self, vertex: GcsVertex, result: MathematicalProgramResult
    ) -> NonCollisionVariables:
        # TODO: This can probably be cleaned up somehow
        p_BF_xs = self._get_vars_solution_for_vertex_vars(vertex.x(), self.variables.p_BP_xs, result)  # type: ignore
        p_BF_ys = self._get_vars_solution_for_vertex_vars(vertex.x(), self.variables.p_BP_ys, result)  # type: ignore
        p_WB_x = self._get_var_solution_for_vertex_vars(vertex.x(), self.variables.p_WB_x, result)  # type: ignore
        p_WB_y = self._get_var_solution_for_vertex_vars(vertex.x(), self.variables.p_WB_y, result)  # type: ignore
        cos_th = self._get_var_solution_for_vertex_vars(vertex.x(), self.variables.cos_th, result)  # type: ignore
        sin_th = self._get_var_solution_for_vertex_vars(vertex.x(), self.variables.sin_th, result)  # type: ignore
        return NonCollisionVariables(
            self.contact_location,
            self.variables.num_knot_points,
            self.variables.time_in_mode,
            self.variables.dt,
            self.dynamics_config.pusher_radius,
            p_BF_xs,
            p_BF_ys,
            p_WB_x,
            p_WB_y,
            cos_th,
            sin_th,
        )

    def get_variable_solutions(
        self, result: MathematicalProgramResult
    ) -> NonCollisionVariables:
        # TODO: This can probably be cleaned up somehow
        p_BF_xs = result.GetSolution(self.variables.p_BP_xs)
        p_BF_ys = result.GetSolution(self.variables.p_BP_ys)
        p_WB_x = result.GetSolution(self.variables.p_WB_x)  # type: ignore
        p_WB_y = result.GetSolution(self.variables.p_WB_y)  # type: ignore
        cos_th = result.GetSolution(self.variables.cos_th)  # type: ignore
        sin_th = result.GetSolution(self.variables.sin_th)  # type: ignore
        return NonCollisionVariables(
            self.contact_location,
            self.variables.num_knot_points,
            self.variables.time_in_mode,
            self.variables.dt,
            self.dynamics_config.pusher_radius,
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

        return poly

    def get_convex_set_in_positions(self) -> opt.Spectrahedron:
        # Construct a small temporary program in R^2 that will allow us to check
        # for positional intersections between regions
        NUM_DIMS = 2
        temp_prog = MathematicalProgram()
        x = temp_prog.NewContinuousVariables(NUM_DIMS, "x")

        exprs = self._create_collision_free_space_constraints(x)
        for e in exprs:
            temp_prog.AddLinearConstraint(e)

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
                self.variables.p_BPs[0],
                self.variables.p_WB,
                self.variables.cos_th,  # type: ignore
                self.variables.sin_th,  # type: ignore
            )
        else:
            return ContinuityVariables(
                self.variables.p_BPs[-1],
                self.variables.p_WB,
                self.variables.cos_th,  # type: ignore
                self.variables.sin_th,  # type: ignore
            )

    # TODO(bernhardpg): refactor common code
    def _get_eucl_dist_cost_term(self) -> Tuple[List[int], QuadraticCost]:
        if self.config.avoid_object and self.config.avoidance_cost == "quadratic":
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
        if self.config.avoidance_cost == "linear":
            assert len(self.prog.linear_costs()) == 1
            object_avoidance_cost = self.prog.linear_costs()[0]
        elif self.config.avoidance_cost == "quadratic":
            assert len(self.prog.quadratic_costs()) == 2
            object_avoidance_cost = self.prog.quadratic_costs()[1]
        else:
            raise NotImplementedError(
                f"Cannot get object avoidance cost terms for cost type {self.config.avoidance_cost}."
            )

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

        if self.config.avoid_object:
            if self.config.avoidance_cost in ["quadratic", "linear"]:
                var_idxs, evaluator = self._get_object_avoidance_cost_term()
                vars = vertex.x()[var_idxs]
                cost_type = (
                    LinearCost
                    if self.config.avoidance_cost == "linear"
                    else QuadraticCost
                )
                binding = Binding[cost_type](evaluator, vars)
                vertex.AddCost(binding)
            else:
                # TODO(bernhardpg): Clean up this part
                planes = self.slider_geometry.get_contact_planes(
                    self.contact_location.idx
                )
                for plane in planes:
                    for p_BP in self.variables.p_BPs:
                        # A = [a^T; 0]
                        NUM_VARS = 2  # num variables required in the PerspectiveQuadraticCost formulation
                        NUM_DIMS = 2
                        A = np.zeros((NUM_VARS, NUM_DIMS))
                        A[0, :] = (
                            plane.a.T
                            * self.config.cost_terms.cost_param_avoidance_socp_weight
                        )
                        # b = [b; 1]
                        b = np.ones((NUM_VARS, 1))
                        b[0] = plane.b
                        b = b * self.config.cost_terms.cost_param_avoidance_socp_weight

                        # z = [a^T x + b; 1]
                        cost = PerspectiveQuadraticCost(A, b)
                        vars_in_vertex = vertex.x()[
                            self.prog.FindDecisionVariableIndices(p_BP)
                        ]
                        vertex.AddCost(
                            Binding[PerspectiveQuadraticCost](cost, vars_in_vertex)
                        )
