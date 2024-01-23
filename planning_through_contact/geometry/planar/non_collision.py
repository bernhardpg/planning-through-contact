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
    L2NormCost,
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
from planning_through_contact.tools.utils import calc_displacements

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
    def v_BPs(self):
        return calc_displacements(self.p_BPs, self.dt)

    @property
    def p_WB(self):
        return np.expand_dims(np.array([self.p_WB_x, self.p_WB_y]), 1)  # (2, 1)

    @property
    def R_WB(self):
        R = np.array([[self.cos_th, -self.sin_th], [self.sin_th, self.cos_th]])
        return R


@dataclass
class NonCollisionMode(AbstractContactMode):
    terminal_cost: bool = False

    @classmethod
    def create_from_plan_spec(
        cls,
        contact_location: PolytopeContactLocation,
        config: PlanarPlanConfig,
        name: Optional[str] = None,
        one_knot_point: bool = False,
        terminal_cost: bool = False,
    ) -> "NonCollisionMode":
        if name is None:
            name = f"NON_COLL_{contact_location.idx}"

        num_knot_points = 1 if one_knot_point else config.num_knot_points_non_collision

        return cls(
            name,
            num_knot_points,
            config.time_non_collision,
            contact_location,
            config,
            terminal_cost,
        )

    @classmethod
    def create_source_or_target_mode(
        cls,
        config: PlanarPlanConfig,
        slider_pose_world: PlanarPose,
        pusher_pose_world: PlanarPose,
        initial_or_final: Literal["initial", "final"],
        set_slider_pose: bool = True,
        terminal_cost: bool = False,
    ) -> "NonCollisionMode":
        p_WP = pusher_pose_world.pos()
        R_WB = slider_pose_world.two_d_rot_matrix()
        p_WB = slider_pose_world.pos()

        # We need to compute the pusher pos in the frame of the slider
        p_BP = R_WB.T @ (p_WP - p_WB)
        pusher_pose_body = PlanarPose(p_BP[0, 0], p_BP[1, 0], 0)

        loc = find_first_matching_location(pusher_pose_body, config)
        mode_name = "source" if initial_or_final == "initial" else "target"
        mode = cls.create_from_plan_spec(
            loc, config, mode_name, one_knot_point=True, terminal_cost=terminal_cost
        )
        if set_slider_pose:
            mode.set_slider_pose(slider_pose_world)

        if initial_or_final == "initial":
            mode.set_finger_initial_pose(pusher_pose_body)
        else:  # final
            mode.set_finger_final_pose(pusher_pose_body)

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
        self.prog = MathematicalProgram()
        self.variables = NonCollisionVariables.from_prog(
            self.prog,
            self.num_knot_points,
            self.time_in_mode,
            self.contact_location,
            self.dynamics_config.pusher_radius,
        )

        self.l2_norm_costs = []
        self.distance_to_object_socp_costs = []
        self.squared_eucl_dist_cost = None
        self.quadratic_distance_cost = None

        self._define_constraints()
        self._define_cost()

    def _define_constraints(self) -> None:
        for k in range(self.num_knot_points):
            p_BF = self.variables.p_BPs[k]

            exprs = self._create_collision_free_space_constraints(p_BF)
            for expr in exprs:
                self.prog.AddLinearConstraint(expr)

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
        if self.num_knot_points == 1 and not self.terminal_cost:
            # If we have only one knot point we are either a source or target mode, in that
            # case we don't add any cost (unless explicitly specified)
            return

        self.cost_config = self.config.non_collision_cost

        if self.cost_config.pusher_velocity_regularization is not None:
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
                self.squared_eucl_dist_cost = self.prog.AddQuadraticCost(
                    self.cost_config.pusher_velocity_regularization * squared_eucl_dist,
                    is_convex=True,
                )

        if self.cost_config.pusher_arc_length is not None:
            for k in range(self.num_knot_points - 1):
                vars = np.concatenate(
                    [
                        self.variables.p_BPs[k].flatten(),
                        self.variables.p_BPs[k + 1].flatten(),
                    ]
                )
                distance = self.variables.p_BPs[k + 1] - self.variables.p_BPs[k]
                cost_expr = self.cost_config.pusher_arc_length * distance
                A, b = sym.DecomposeAffineExpressions(cost_expr, vars)
                cost = self.prog.AddL2NormCost(A, b, vars)
                self.l2_norm_costs.append(cost)

        if self.cost_config.avoid_object:
            planes = self.slider_geometry.get_contact_planes(self.contact_location.idx)
            dists_for_each_plane = [
                [plane.dist_to(p_BF) for p_BF in self.variables.p_BPs]
                for plane in planes
            ]

            if self.cost_config.distance_to_object_quadratic is not None:
                squared_dists = [
                    self.cost_config.distance_to_object_quadratic
                    * (
                        d
                        - self.cost_config.distance_to_object_quadratic_preferred_distance
                    )
                    ** 2
                    for dist in dists_for_each_plane
                    for d in dist
                ]
                self.quadratic_distance_cost = self.prog.AddQuadraticCost(
                    np.sum(squared_dists), is_convex=True
                )

            if self.cost_config.distance_to_object_socp is not None:
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
                        A[0, :] = plane.a.T * self.cost_config.distance_to_object_socp
                        # b = [b; 1]
                        b = np.ones((NUM_VARS, 1))
                        b[0] = plane.b
                        b = b * self.cost_config.distance_to_object_socp

                        # z = [a^T x + b; 1]
                        cost = PerspectiveQuadraticCost(A, b)
                        binding = self.prog.AddCost(cost, p_BP)
                        self.distance_to_object_socp_costs.append(binding)

            if self.cost_config.distance_to_object_socp_single_mode is not None:
                # TODO: Can probably get rid of this

                # NOTE: Only a research feature for adding socp costs on a single mode (i.e. not in GCS)
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
                            self.cost_config.distance_to_object_socp_single_mode * s
                        )

        # TODO: This is only used with ContactCost.OPTIMAL_CONTROL, and can be removed
        if self.terminal_cost:  # Penalize difference from target position on slider.
            assert self.config.start_and_goal is not None

            pos_diff = (
                self.variables.p_WB
                - self.config.start_and_goal.slider_target_pose.pos()
            )
            self.terminal_cost_pos = self.prog.AddQuadraticCost((pos_diff.T @ pos_diff).item())  # type: ignore

            th = self.config.start_and_goal.slider_target_pose.theta
            cos_th_target = np.cos(th)
            sin_th_target = np.sin(th)

            rot_diff = (self.variables.cos_th - cos_th_target) ** 2 + (
                self.variables.sin_th - sin_th_target
            ) ** 2
            self.terminal_cost_rot = self.prog.AddQuadraticCost(rot_diff)  # type: ignore

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

        # Sets should automatically be bounded as we have to touch the object to move it
        # GCS requires the sets to be bounded
        # TODO: Currently, some tests will fail if this is not enabled (as the sets are unbounded)
        # TODO: This will be handled soon
        if make_bounded:
            BOUND = 2  # TODO(bernhardpg): this should not be hardcoded
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
                # There will be no velocities if there is only one knot point
                # (i.e. for source or target vertex)
                self.variables.v_BPs[0] if self.num_knot_points > 1 else None,
            )
        else:
            return ContinuityVariables(
                self.variables.p_BPs[-1],
                self.variables.p_WB,
                self.variables.cos_th,  # type: ignore
                self.variables.sin_th,  # type: ignore
                # There will be no velocities if there is only one knot point
                # (i.e. for source or target vertex)
                self.variables.v_BPs[-1] if self.num_knot_points > 1 else None,
            )

    def _get_cost_terms(self, cost: Binding) -> Tuple[List[int], QuadraticCost]:
        var_idxs = self.get_variable_indices_in_gcs_vertex(cost.variables())
        return var_idxs, cost.evaluator()

    def add_cost_to_vertex(self, vertex: GcsVertex) -> None:
        is_target_or_source = self.num_knot_points == 1

        # TODO: This is old code that is only used with ContactCost.OPTIMAL_CONTROL, and can be removed
        if is_target_or_source:
            assert (
                len(self.prog.quadratic_costs()) == 2
            )  # should be one cost for pos and one for rot
            assert self.terminal_cost_pos is not None
            assert self.terminal_cost_rot is not None

            for binding in (self.terminal_cost_pos, self.terminal_cost_rot):
                var_idxs = self.get_variable_indices_in_gcs_vertex(binding.variables())
                vars = vertex.x()[var_idxs]
                new_binding = Binding[QuadraticCost](binding.evaluator(), vars)
                vertex.AddCost(new_binding)
        else:
            if self.cost_config.pusher_velocity_regularization is not None:
                var_idxs, evaluator = self._get_cost_terms(self.squared_eucl_dist_cost)
                vars = vertex.x()[var_idxs]
                binding = Binding[QuadraticCost](evaluator, vars)
                vertex.AddCost(binding)

            if self.cost_config.pusher_arc_length is not None:
                assert len(self.l2_norm_costs) > 0

                # Add L2 norm cost terms
                for cost in self.l2_norm_costs:
                    var_idxs, evaluator = self._get_cost_terms(cost)
                    vars = vertex.x()[var_idxs]
                    binding = Binding[L2NormCost](evaluator, vars)
                    vertex.AddCost(binding)

            if self.cost_config.distance_to_object_quadratic is not None:
                var_idxs, evaluator = self._get_cost_terms(self.quadratic_distance_cost)
                vars = vertex.x()[var_idxs]
                binding = Binding[QuadraticCost](evaluator, vars)
                vertex.AddCost(binding)

            if self.cost_config.distance_to_object_socp:
                for binding in self.distance_to_object_socp_costs:
                    var_idxs, evaluator = self._get_cost_terms(binding)
                    vars = vertex.x()[var_idxs]
                    binding = Binding[PerspectiveQuadraticCost](evaluator, vars)
                    vertex.AddCost(binding)
