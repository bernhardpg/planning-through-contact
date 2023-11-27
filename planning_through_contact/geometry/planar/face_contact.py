from dataclasses import dataclass
from itertools import combinations
from typing import Callable, List, Literal, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pydrake.geometry.optimization as opt
import pydrake.symbolic as sym
from pydrake.math import eq
from pydrake.solvers import (
    Binding,
    LinearCost,
    MakeSemidefiniteRelaxation,
    MathematicalProgram,
    MathematicalProgramResult,
)

from planning_through_contact.convex_relaxation.sdp import (
    eliminate_equality_constraints,
)
from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    CollisionGeometry,
    PolytopeContactLocation,
)
from planning_through_contact.geometry.planar.abstract_mode import (
    AbstractContactMode,
    AbstractModeVariables,
    ContinuityVariables,
)
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.rigid_body import RigidBody
from planning_through_contact.geometry.utilities import cross_2d
from planning_through_contact.planning.planar.planar_plan_config import (
    PlanarPlanConfig,
    SliderPusherSystemConfig,
)
from planning_through_contact.tools.types import NpExpressionArray, NpVariableArray
from planning_through_contact.tools.utils import (
    approx_exponential_map,
    calc_displacements,
    skew_symmetric_so2,
)

GcsVertex = opt.GraphOfConvexSets.Vertex
GcsEdge = opt.GraphOfConvexSets.Edge


@dataclass
class FaceContactVariables(AbstractModeVariables):
    lams: NpVariableArray | npt.NDArray[np.float64]  # (num_knot_points, )
    normal_forces: NpVariableArray | npt.NDArray[np.float64]  # (num_knot_points, )
    friction_forces: NpVariableArray | npt.NDArray[np.float64]  # (num_knot_points, )
    cos_ths: NpVariableArray | npt.NDArray[np.float64]  # (num_knot_points, )
    sin_ths: NpVariableArray | npt.NDArray[np.float64]  # (num_knot_points, )
    theta_dots: Optional[
        NpVariableArray | npt.NDArray[np.float64]
    ]  # (num_knot_points, )
    p_WB_xs: NpVariableArray | npt.NDArray[np.float64]  # (num_knot_points, )
    p_WB_ys: NpVariableArray | npt.NDArray[np.float64]  # (num_knot_points, )

    pv1: npt.NDArray[np.float64]
    pv2: npt.NDArray[np.float64]
    normal_vec: npt.NDArray[np.float64]
    tangent_vec: npt.NDArray[np.float64]

    @classmethod
    def from_prog(
        cls,
        prog: MathematicalProgram,
        object_geometry: CollisionGeometry,
        contact_location: PolytopeContactLocation,
        num_knot_points: int,
        time_in_mode: float,
        pusher_radius: float,
        define_theta_dots: bool = True,
    ) -> "FaceContactVariables":
        # Contact positions
        rel_finger_pos = prog.NewContinuousVariables(num_knot_points, "rel_finger_pos")
        pv1, pv2 = object_geometry.get_proximate_vertices_from_location(
            contact_location
        )

        # Contact forces
        num_inputs = num_knot_points - 1
        normal_forces = prog.NewContinuousVariables(num_inputs, "c_n")
        friction_forces = prog.NewContinuousVariables(num_inputs, "c_f")
        (
            normal_vec,
            tangent_vec,
        ) = object_geometry.get_norm_and_tang_vecs_from_location(contact_location)
        if define_theta_dots:
            theta_dots = prog.NewContinuousVariables(num_inputs, "theta_dot")
        else:
            theta_dots = None

        # Rotations
        cos_ths = prog.NewContinuousVariables(num_knot_points, "cos_th")
        sin_ths = prog.NewContinuousVariables(num_knot_points, "sin_th")

        # Box position relative to world frame
        p_WB_xs = prog.NewContinuousVariables(num_knot_points, "p_WB_x")
        p_WB_ys = prog.NewContinuousVariables(num_knot_points, "p_WB_y")

        dt = time_in_mode / num_knot_points  # TODO: Remove

        return FaceContactVariables(
            contact_location,
            num_knot_points,
            time_in_mode,  # TODO: Remove
            dt,  # TODO: Remove
            pusher_radius,
            rel_finger_pos,
            normal_forces,
            friction_forces,
            cos_ths,
            sin_ths,
            theta_dots,
            p_WB_xs,
            p_WB_ys,
            pv1,
            pv2,
            normal_vec,
            tangent_vec,
        )

    def eval_result(self, result: MathematicalProgramResult) -> "FaceContactVariables":
        def get_float_from_result(
            vec: npt.NDArray[np.float64] | NpExpressionArray,
        ) -> npt.NDArray[np.float64]:
            if vec.dtype == np.float64:
                return result.GetSolution(vec)
            elif vec.dtype == np.object_:
                return sym.Evaluate(result.GetSolution(vec)).flatten()  # type: ignore
            else:
                raise NotImplementedError(f"dtype {vec.dtype} not supported")

        return FaceContactVariables(
            self.contact_location,
            self.num_knot_points,
            self.time_in_mode,
            self.dt,
            self.pusher_radius,
            get_float_from_result(self.lams),
            get_float_from_result(self.normal_forces),
            get_float_from_result(self.friction_forces),
            get_float_from_result(self.cos_ths),
            get_float_from_result(self.sin_ths),
            get_float_from_result(self.theta_dots)
            if self.theta_dots is not None
            else None,
            get_float_from_result(self.p_WB_xs),
            get_float_from_result(self.p_WB_ys),
            self.pv1,
            self.pv2,
            self.normal_vec,
            self.tangent_vec,
        )

    def from_reduced_prog(
        self,
        original_prog: MathematicalProgram,
        reduced_prog: MathematicalProgram,
        get_original_exprs: Callable,
    ) -> "FaceContactVariables":
        original_as_expressions = get_original_exprs(reduced_prog.decision_variables())

        get_original_vars_from_reduced = lambda original_vars: original_as_expressions[
            original_prog.FindDecisionVariableIndices(original_vars)
        ].flatten()

        return FaceContactVariables(
            self.contact_location,
            self.num_knot_points,
            self.time_in_mode,
            self.dt,
            self.pusher_radius,
            get_original_vars_from_reduced(self.lams),
            get_original_vars_from_reduced(self.normal_forces),
            get_original_vars_from_reduced(self.friction_forces),
            get_original_vars_from_reduced(self.cos_ths),
            get_original_vars_from_reduced(self.sin_ths),
            get_original_vars_from_reduced(self.theta_dots)
            if self.theta_dots is not None
            else None,
            get_original_vars_from_reduced(self.p_WB_xs),
            get_original_vars_from_reduced(self.p_WB_ys),
            self.pv1,
            self.pv2,
            self.normal_vec,
            self.tangent_vec,
        )

    @property
    def R_WBs(self):
        Rs = [
            np.array([[cos, -sin], [sin, cos]])
            for cos, sin in zip(self.cos_ths, self.sin_ths)
        ]
        return Rs

    @property
    def p_WBs(self):
        return [
            np.array([x, y]).reshape((2, 1)) for x, y in zip(self.p_WB_xs, self.p_WB_ys)
        ]

    @property
    def f_c_Bs(self):
        return [
            c_n * self.normal_vec + c_f * self.tangent_vec
            for c_n, c_f in zip(self.normal_forces, self.friction_forces)
        ]

    @property
    def omega_hats(self):
        assert self.theta_dots is not None
        return [skew_symmetric_so2(th_dot) for th_dot in self.theta_dots]

    def _get_p_BP(self, lam: float, pusher_radius: float):
        point_on_surface = lam * self.pv1 + (1 - lam) * self.pv2
        radius_displacement = -self.normal_vec * pusher_radius
        return point_on_surface + radius_displacement

    @property
    def p_BPs(self):
        return [self._get_p_BP(lam, self.pusher_radius) for lam in self.lams]

    @property
    def v_WBs(self):
        return calc_displacements(self.p_WBs)

    @property
    def v_BPs(self):
        return calc_displacements(self.p_BPs)

    @property
    def p_WPs(self):
        return [
            p_WB + R_WB.dot(p_BP)
            for p_WB, R_WB, p_BP in zip(self.p_WBs, self.R_WBs, self.p_BPs)
        ]

    @property
    def f_c_Ws(self):
        return [R_WB.dot(f_c_B) for f_c_B, R_WB in zip(self.f_c_Bs, self.R_WBs)]

    @property
    def delta_cos_ths(self):
        return np.array(calc_displacements(self.cos_ths))

    @property
    def delta_sin_ths(self):
        return np.array(calc_displacements(self.sin_ths))

    @property
    def delta_omega_WBs(self):
        delta_R_WBs = [
            np.array([[cos_dot, -sin_dot], [sin_dot, cos_dot]])
            for cos_dot, sin_dot in zip(self.delta_cos_ths, self.delta_sin_ths)
        ]
        # In 2D, omega_z = theta_dot will be at position (1,0) in R_dot * R'
        oms = [R_dot.dot(R.T)[1, 0] for R, R_dot in zip(self.R_WBs, delta_R_WBs)]
        return oms


@dataclass
class FaceContactMode(AbstractContactMode):
    @classmethod
    def create_from_plan_spec(
        cls,
        contact_location: PolytopeContactLocation,
        config: PlanarPlanConfig,
    ) -> "FaceContactMode":
        prog = MathematicalProgram()
        name = str(contact_location)
        return cls(
            name,
            config.num_knot_points_contact,
            config.time_in_contact,
            contact_location,
            prog,
            config,
        )

    def __post_init__(self) -> None:
        self.dynamics_config = self.config.dynamics_config
        self.relaxed_prog = None
        self.variables = FaceContactVariables.from_prog(
            self.prog,
            self.dynamics_config.slider.geometry,
            self.contact_location,
            self.num_knot_points,
            self.time_in_mode,
            self.dynamics_config.pusher_radius,
            define_theta_dots=self.config.use_approx_exponential_map,
        )
        self.constraints = {
            "SO2": [],
            "rotational_dynamics": [],
            "translational_dynamics": [],
            "translational_dynamics_red": [],
        }
        if self.config.use_approx_exponential_map:
            self.constraints["exponential_map"] = []
        self._define_constraints()
        self._define_costs()

    def _define_constraints(self) -> None:
        for lam in self.variables.lams:
            self.prog.AddBoundingBoxConstraint(0, 1, lam)

        # SO(2) constraints
        for c, s in zip(self.variables.cos_ths, self.variables.sin_ths):
            constraint = c**2 + s**2 - 1
            self.prog.AddQuadraticConstraint(constraint, 0, 0)
            self.constraints["SO2"].append(constraint)

        if self.config.use_approx_exponential_map:
            # so(2) (tangent space) constraints
            for k in range(self.num_knot_points - 1):
                R_k = self.variables.R_WBs[k]
                R_k_next = self.variables.R_WBs[k + 1]
                omega_hat_k = self.variables.omega_hats[k]
                exp_map = approx_exponential_map(omega_hat_k)

                # NOTE: For now we only add the one side of the exp map constraint
                constraint = exp_map - R_k.T @ R_k_next
                self.constraints["exponential_map"].append(constraint.flatten())
                for c in constraint.flatten():
                    self.prog.AddQuadraticConstraint(c, 0, 0)

        # Friction cone constraints
        for c_n in self.variables.normal_forces:
            self.prog.AddBoundingBoxConstraint(0, np.inf, c_n)

        mu = self.dynamics_config.friction_coeff_slider_pusher
        for c_n, c_f in zip(
            self.variables.normal_forces, self.variables.friction_forces
        ):
            self.prog.AddLinearConstraint(c_f <= mu * c_n)
            self.prog.AddLinearConstraint(c_f >= -mu * c_n)

        # These position bounds should never be needed, as either the initial position,
        # target position, or edge constraints with this mode (in the case of GCS)
        # will constrain them
        self.bound_positions = False
        if self.bound_positions:
            lb, ub = self.config.workspace.slider.bounds
            for p_WB in self.variables.p_WBs:
                self.prog.AddBoundingBoxConstraint(lb, ub, p_WB)

        # Bounds on cosines and sines
        for cos_th, sin_th in zip(self.variables.cos_ths, self.variables.sin_ths):
            self.prog.AddBoundingBoxConstraint(-1, 1, cos_th)
            self.prog.AddBoundingBoxConstraint(-1, 1, sin_th)

        # Quasi-static dynamics
        for k in range(self.num_knot_points - 1):
            v_WB = self.variables.v_WBs[k]

            if self.config.use_approx_exponential_map:
                assert self.variables.theta_dots is not None
                theta_WB_dot = self.variables.theta_dots[k]
            else:
                theta_WB_dot = self.variables.delta_omega_WBs[k]

            f_c_B = self.variables.f_c_Bs[k]
            p_BP = self.variables.p_BPs[k]
            R_WB = self.variables.R_WBs[k]

            trans_vel_constraint = v_WB - R_WB @ f_c_B
            self.constraints["translational_dynamics"].append(
                trans_vel_constraint.flatten()
            )
            for c in trans_vel_constraint.flatten():
                self.prog.AddQuadraticConstraint(c, 0, 0)

            trans_vel_constraint = R_WB.T @ v_WB - f_c_B
            self.constraints["translational_dynamics_red"].append(
                trans_vel_constraint.flatten()
            )
            for c in trans_vel_constraint.flatten():
                self.prog.AddQuadraticConstraint(c, 0, 0)

            c = self.config.dynamics_config.limit_surface_const
            ang_vel_constraint = theta_WB_dot - c * cross_2d(p_BP, f_c_B)
            self.constraints["rotational_dynamics"].append(ang_vel_constraint)
            self.prog.AddQuadraticConstraint(ang_vel_constraint, 0, 0)

        c = self.config.dynamics_config.limit_surface_const
        for k, l in combinations(range(self.num_knot_points - 1), 2):
            assert self.variables.theta_dots is not None
            th1 = self.variables.theta_dots[k]
            f1 = self.variables.f_c_Bs[k]
            p1 = self.variables.p_BPs[k]
            th2 = self.variables.theta_dots[l]
            f2 = self.variables.f_c_Bs[l]
            p2 = self.variables.p_BPs[l]

            const = (
                c * (cross_2d(p1 - p2, f1 - f2) + cross_2d(p1, f2) + cross_2d(p2, f1))
                - th1
                - th2
            )
            self.prog.AddQuadraticConstraint(const, 0, 0)

        # Ensure sticking on the contact point
        for v_c_B in self.variables.v_BPs:
            self.prog.AddLinearEqualityConstraint(v_c_B.flatten(), np.zeros((2,)))

    def _define_costs(self) -> None:
        # Minimize kinetic energy through squared velocities
        sq_linear_vels = sum([v_WB.T.dot(v_WB) for v_WB in self.variables.v_WBs]).item()  # type: ignore
        self.prog.AddQuadraticCost(
            self.config.cost_terms.cost_param_lin_vels * sq_linear_vels
        )

        # TODO(bernhardpg): Remove
        if self.config.use_approx_exponential_map:
            sq_angular_vels = self.variables.theta_dots @ self.variables.theta_dots  # type: ignore
            self.prog.AddQuadraticCost(self.config.cost_terms.cost_param_ang_vels * sq_angular_vels)  # type: ignore
        else:
            self.prog.AddQuadraticCost(
                self.variables.delta_cos_ths.T @ self.variables.delta_cos_ths  # type: ignore
                + self.variables.delta_sin_ths.T @ self.variables.delta_sin_ths
            )
        #
        # sq_normal_forces = self.variables.normal_forces @ self.variables.normal_forces  # type: ignore
        # self.prog.AddQuadraticCost(self.config.cost_terms.cost_param_forces * sq_normal_forces)  # type: ignore
        #
        # sq_friction_forces = self.variables.friction_forces @ self.variables.friction_forces  # type: ignore
        # self.prog.AddQuadraticCost(self.config.cost_terms.cost_param_forces * sq_friction_forces)  # type: ignore

    def set_finger_pos(self, lam_target: float) -> None:
        """
        Set finger position along the contact face.
        As the finger position is constant, there is no difference between
        initial and target value.

        @param lam_target: Position along face, value 0 to 1.
        """
        if lam_target >= 1 or lam_target <= 0:
            raise ValueError("The finger position should be set between 0 and 1")

        # for lam in self.variables.lams:
        #     self.prog.AddLinearConstraint(lam >= lam_target)
        #     self.prog.AddLinearConstraint(lam <= lam_target)

        self.prog.AddLinearConstraint(
            eq(self.variables.lams, np.full(self.variables.lams.shape, lam_target))
        )

    def set_slider_initial_pose(self, pose: PlanarPose) -> None:
        self.prog.AddLinearConstraint(self.variables.cos_ths[0] == np.cos(pose.theta))
        self.prog.AddLinearConstraint(self.variables.sin_ths[0] == np.sin(pose.theta))
        self.prog.AddLinearConstraint(eq(self.variables.p_WBs[0], pose.pos()))

        self.slider_initial_pose = pose

    def set_slider_final_pose(self, pose: PlanarPose) -> None:
        self.prog.AddLinearConstraint(self.variables.cos_ths[-1] == np.cos(pose.theta))
        self.prog.AddLinearConstraint(self.variables.sin_ths[-1] == np.sin(pose.theta))
        self.prog.AddLinearConstraint(eq(self.variables.p_WBs[-1], pose.pos()))

        self.slider_final_pose = pose

    def formulate_convex_relaxation(self) -> None:
        if self.config.use_eq_elimination:
            self.original_prog = self.prog
            (
                self.reduced_prog,
                self.get_original_vars_from_reduced,
            ) = eliminate_equality_constraints(self.prog)
            self.prog = self.reduced_prog

            # TODO(bernhardpg): Clean up this
            self.original_variables = self.variables
            self.variables = self.variables.from_reduced_prog(
                self.original_prog,
                self.reduced_prog,
                self.get_original_vars_from_reduced,
            )
            c = self.reduced_prog.linear_constraints()[0]

            self.relaxed_prog = MakeSemidefiniteRelaxation(self.reduced_prog)
        else:
            self.relaxed_prog = MakeSemidefiniteRelaxation(self.prog)

    def get_convex_set(self) -> opt.Spectrahedron:
        if self.relaxed_prog is None:
            self.formulate_convex_relaxation()

        return opt.Spectrahedron(self.relaxed_prog)

    def get_variable_indices_in_gcs_vertex(self, vars: NpVariableArray) -> List[int]:
        return self.prog.FindDecisionVariableIndices(vars)
        # NOTE: This function relies on the fact that the sdp relaxation
        # returns an ordering of variables [1, x1, x2, ...],
        # where [x1, x2, ...] is the original ordering in self.prog

    def get_variable_solutions_for_vertex(
        self, vertex: GcsVertex, result: MathematicalProgramResult
    ) -> FaceContactVariables:
        # TODO: This can probably be cleaned up somehow
        lams = self._get_vars_solution_for_vertex_vars(vertex.x(), self.variables.lams, result)  # type: ignore
        normal_forces = self._get_vars_solution_for_vertex_vars(
            vertex.x(), self.variables.normal_forces, result  # type: ignore
        )
        friction_forces = self._get_vars_solution_for_vertex_vars(
            vertex.x(), self.variables.friction_forces, result  # type: ignore
        )
        cos_ths = self._get_vars_solution_for_vertex_vars(vertex.x(), self.variables.cos_ths, result)  # type: ignore
        sin_ths = self._get_vars_solution_for_vertex_vars(vertex.x(), self.variables.sin_ths, result)  # type: ignore
        if self.config.use_approx_exponential_map:
            theta_dots = self._get_vars_solution_for_vertex_vars(vertex.x(), self.variables.theta_dots, result)  # type: ignore
        else:
            theta_dots = None
        p_WB_xs = self._get_vars_solution_for_vertex_vars(vertex.x(), self.variables.p_WB_xs, result)  # type: ignore
        p_WB_ys = self._get_vars_solution_for_vertex_vars(vertex.x(), self.variables.p_WB_ys, result)  # type: ignore

        return FaceContactVariables(
            self.contact_location,
            self.variables.num_knot_points,
            self.variables.time_in_mode,
            self.variables.dt,
            self.dynamics_config.pusher_radius,
            lams,
            normal_forces,
            friction_forces,
            cos_ths,
            sin_ths,
            theta_dots,
            p_WB_xs,
            p_WB_ys,
            self.variables.pv1,
            self.variables.pv2,
            self.variables.normal_vec,
            self.variables.tangent_vec,
        )

    def get_variable_solutions(
        self, result: MathematicalProgramResult
    ) -> FaceContactVariables:
        # TODO: This can probably be cleaned up somehow
        lams = result.GetSolution(self.variables.lams)
        normal_forces = result.GetSolution(self.variables.normal_forces)
        friction_forces = result.GetSolution(self.variables.friction_forces)
        cos_ths = result.GetSolution(self.variables.cos_ths)  # type: ignore
        sin_ths = result.GetSolution(self.variables.sin_ths)  # type: ignore
        theta_dots = result.GetSolution(self.variables.theta_dots)  # type: ignore
        p_WB_xs = result.GetSolution(self.variables.p_WB_xs)  # type: ignore
        p_WB_ys = result.GetSolution(self.variables.p_WB_ys)  # type: ignore

        return FaceContactVariables(
            self.contact_location,
            self.variables.num_knot_points,
            self.variables.time_in_mode,
            self.variables.dt,
            self.dynamics_config.pusher_radius,
            lams,
            normal_forces,
            friction_forces,
            cos_ths,
            sin_ths,
            theta_dots,
            p_WB_xs,
            p_WB_ys,
            self.variables.pv1,
            self.variables.pv2,
            self.variables.normal_vec,
            self.variables.tangent_vec,
        )

    def get_continuity_vars(
        self, first_or_last: Literal["first", "last"]
    ) -> ContinuityVariables:
        if first_or_last == "first":
            return ContinuityVariables(
                self.variables.p_BPs[0],  # type: ignore
                self.variables.p_WBs[0],
                self.variables.cos_ths[0],
                self.variables.sin_ths[0],
            )
        else:
            return ContinuityVariables(
                self.variables.p_BPs[-1],  # type: ignore
                self.variables.p_WBs[-1],
                self.variables.cos_ths[-1],
                self.variables.sin_ths[-1],
            )

    def _get_cost_terms(self) -> Tuple[List[List[int]], List[LinearCost]]:
        if self.relaxed_prog is None:
            raise RuntimeError(
                "Relaxed program must be constructed before cost can be formulated for vertex."
            )

        costs = self.relaxed_prog.linear_costs()
        evaluators = [cost.evaluator() for cost in costs]
        # NOTE: here we must get the indices from the relaxed program!
        var_idxs = [
            self.relaxed_prog.FindDecisionVariableIndices(cost.variables())
            for cost in costs
        ]
        return var_idxs, evaluators

    def add_cost_to_vertex(self, vertex: GcsVertex) -> None:
        var_idxs, evaluators = self._get_cost_terms()
        vars = [vertex.x()[idxs] for idxs in var_idxs]
        bindings = [Binding[LinearCost](e, v) for e, v in zip(evaluators, vars)]
        for b in bindings:
            vertex.AddCost(b)
