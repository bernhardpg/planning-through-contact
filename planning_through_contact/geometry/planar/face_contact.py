from dataclasses import dataclass
from typing import Callable, List, Literal, Tuple

import numpy as np
import numpy.typing as npt
import pydrake.geometry.optimization as opt
from pydrake.math import eq
from pydrake.solvers import (
    Binding,
    LinearCost,
    MakeSemidefiniteRelaxation,
    MathematicalProgram,
    MathematicalProgramResult,
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
from planning_through_contact.planning.planar.planar_plan_specs import PlanarPlanSpecs
from planning_through_contact.tools.types import NpVariableArray
from planning_through_contact.tools.utils import forward_differences

GcsVertex = opt.GraphOfConvexSets.Vertex
GcsEdge = opt.GraphOfConvexSets.Edge


@dataclass
class FaceContactVariables(AbstractModeVariables):
    lams: NpVariableArray | npt.NDArray[np.float64]  # (num_knot_points, )
    normal_forces: NpVariableArray | npt.NDArray[np.float64]  # (num_knot_points, )
    friction_forces: NpVariableArray | npt.NDArray[np.float64]  # (num_knot_points, )
    cos_ths: NpVariableArray | npt.NDArray[np.float64]  # (num_knot_points, )
    sin_ths: NpVariableArray | npt.NDArray[np.float64]  # (num_knot_points, )
    p_WB_xs: NpVariableArray | npt.NDArray[np.float64]  # (num_knot_points, )
    p_WB_ys: NpVariableArray | npt.NDArray[np.float64]  # (num_knot_points, )
    omega_WBs: NpVariableArray | npt.NDArray[np.float64]  # (num_knot_points, )

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
    ) -> "FaceContactVariables":
        # Contact positions
        lams = prog.NewContinuousVariables(num_knot_points, "lam")
        pv1, pv2 = object_geometry.get_proximate_vertices_from_location(
            contact_location
        )

        # Contact forces
        normal_forces = prog.NewContinuousVariables(num_knot_points, "c_n")
        friction_forces = prog.NewContinuousVariables(num_knot_points, "c_f")
        (
            normal_vec,
            tangent_vec,
        ) = object_geometry.get_norm_and_tang_vecs_from_location(contact_location)

        # Rotations
        cos_ths = prog.NewContinuousVariables(num_knot_points, "cos_th")
        sin_ths = prog.NewContinuousVariables(num_knot_points, "sin_th")

        # Box position relative to world frame
        p_WB_xs = prog.NewContinuousVariables(num_knot_points, "p_WB_x")
        p_WB_ys = prog.NewContinuousVariables(num_knot_points, "p_WB_y")

        # Angular velocity
        omega_WBs = prog.NewContinuousVariables(num_knot_points - 1, "omega_WB")

        dt = time_in_mode / num_knot_points

        return FaceContactVariables(
            num_knot_points,
            time_in_mode,
            dt,
            lams,
            normal_forces,
            friction_forces,
            cos_ths,
            sin_ths,
            p_WB_xs,
            p_WB_ys,
            omega_WBs,
            pv1,
            pv2,
            normal_vec,
            tangent_vec,
        )

    def eval_result(self, result: MathematicalProgramResult) -> "FaceContactVariables":
        return FaceContactVariables(
            self.num_knot_points,
            self.time_in_mode,
            self.dt,
            result.GetSolution(self.lams),
            result.GetSolution(self.normal_forces),
            result.GetSolution(self.friction_forces),
            result.GetSolution(self.cos_ths),
            result.GetSolution(self.sin_ths),
            result.GetSolution(self.p_WB_xs),
            result.GetSolution(self.p_WB_ys),
            result.GetSolution(self.omega_WBs),
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
    def p_c_Bs(self):
        return [lam * self.pv1 + (1 - lam) * self.pv2 for lam in self.lams]

    @property
    def v_WBs(self):
        return forward_differences(self.p_WBs, self.dt)

    @property
    def cos_th_dots(self):
        return forward_differences(self.cos_ths, self.dt)

    @property
    def sin_th_dots(self):
        return forward_differences(self.sin_ths, self.dt)

    @property
    def v_c_Bs(self):
        return forward_differences(
            self.p_c_Bs, self.dt
        )  # NOTE: Not real velocity, only time differentiation of coordinates (not equal as B is not an inertial frame)!

    # TODO(bernhardpg): Remove this
    # @property
    # def omega_WBs(self):
    #     R_WB_dots = [
    #         np.array([[cos_dot, -sin_dot], [sin_dot, cos_dot]])
    #         for cos_dot, sin_dot in zip(self.cos_th_dots, self.sin_th_dots)
    #     ]
    #     # In 2D, omega_z = theta_dot will be at position (1,0) in R_dot * R'
    #     oms = [R_dot.dot(R.T)[1, 0] for R, R_dot in zip(self.R_WBs, R_WB_dots)]
    #     return oms

    @property
    def R_WB_dots(self):
        return [
            np.array([[cos_dot, -sin_dot], [sin_dot, cos_dot]])
            for cos_dot, sin_dot in zip(self.cos_th_dots, self.sin_th_dots)
        ]

    @property
    def p_c_Ws(self):
        return [
            p_WB + R_WB.dot(p_c_B)
            for p_WB, R_WB, p_c_B in zip(self.p_WBs, self.R_WBs, self.p_c_Bs)
        ]

    @property
    def f_c_Ws(self):
        return [R_WB.dot(f_c_B) for f_c_B, R_WB in zip(self.f_c_Bs, self.R_WBs)]


@dataclass
class FaceContactMode(AbstractContactMode):
    cost_param_lin_vels: float = 1.0
    cost_param_ang_vels: float = 1.0

    @classmethod
    def create_from_plan_spec(
        cls,
        contact_location: PolytopeContactLocation,
        specs: PlanarPlanSpecs,
        object: RigidBody,
    ) -> "FaceContactMode":
        prog = MathematicalProgram()
        name = str(contact_location)
        return cls(
            name,
            specs.num_knot_points_contact,
            specs.time_in_contact,
            contact_location,
            object,
            prog,
        )

    def __post_init__(self) -> None:
        self.relaxed_prog = None
        self.variables = FaceContactVariables.from_prog(
            self.prog,
            self.object.geometry,
            self.contact_location,
            self.num_knot_points,
            self.time_in_mode,
        )
        # TODO(bernhardpg): Should we use this?
        self.enforce_equal_forces = False

        self._define_constraints()
        self._define_costs()

    def _define_constraints(self) -> None:
        # TODO: take this from drake simulation
        FRICTION_COEFF = 0.5
        G = 9.81
        force_max = FRICTION_COEFF * self.object.mass * G
        TABLE_SIZE = 1.0

        for lam in self.variables.lams:
            self.prog.AddBoundingBoxConstraint(0, 1, lam)

        # SO(2) constraints
        for c, s in zip(self.variables.cos_ths, self.variables.sin_ths):
            self.prog.AddConstraint(c**2 + s**2 == 1)

        # Friction cone constraints
        for c_n in self.variables.normal_forces:
            self.prog.AddBoundingBoxConstraint(0, force_max, c_n)

        # TODO(bernhardpg): Compute f_max and tau_max correctly
        # torque_max = force_max * self.object.geometry.get_max_contact_arm(
        #     self.contact_location
        # )
        torque_max = force_max * 0.6 * 0.2
        # Friction cone constraints
        # for omega_WB in self.variables.omega_WBs:
        # self.prog.AddBoundingBoxConstraint(
        #     -1 / torque_max, 1 / torque_max, omega_WB
        # )
        # self.prog.AddBoundingBoxConstraint(-0.3, 0.3, omega_WB)

        for c_n, c_f in zip(
            self.variables.normal_forces, self.variables.friction_forces
        ):
            self.prog.AddLinearConstraint(c_f <= FRICTION_COEFF * c_n)
            self.prog.AddLinearConstraint(c_f >= -FRICTION_COEFF * c_n)

        # Bounds on forces
        for c_n, c_f in zip(
            self.variables.normal_forces, self.variables.friction_forces
        ):
            self.prog.AddBoundingBoxConstraint(-force_max, force_max, c_f)

        # Bounds on positions
        for p_WB_x, p_WB_y in zip(self.variables.p_WB_xs, self.variables.p_WB_ys):
            self.prog.AddBoundingBoxConstraint(-TABLE_SIZE / 2, TABLE_SIZE / 2, p_WB_x)
            self.prog.AddBoundingBoxConstraint(-TABLE_SIZE / 2, TABLE_SIZE / 2, p_WB_y)

        # Bounds on cosines and sines
        for cos_th, sin_th in zip(self.variables.cos_ths, self.variables.sin_ths):
            self.prog.AddBoundingBoxConstraint(-1, 1, cos_th)
            self.prog.AddBoundingBoxConstraint(-1, 1, sin_th)

        if self.enforce_equal_forces:
            # Enforces forces are constant
            for c_n_curr, c_n_next in zip(
                self.variables.normal_forces[:-1], self.variables.normal_forces[1:]
            ):
                self.prog.AddLinearConstraint(c_n_curr == c_n_next)
            for c_f_curr, c_f_next in zip(
                self.variables.friction_forces[:-1], self.variables.friction_forces[1:]
            ):
                self.prog.AddLinearConstraint(c_f_curr == c_f_next)

        # Quasi-static dynamics
        use_midpoint = True
        for k in range(self.num_knot_points - 1):
            v_WB = self.variables.v_WBs[k]
            omega_WB = self.variables.omega_WBs[k]

            # NOTE: We enforce dynamics at midway points as this is where the velocity is 'valid'
            if use_midpoint:
                f_c_B = self._get_midpoint(self.variables.f_c_Bs, k)
                p_c_B = self._get_midpoint(self.variables.p_c_Bs, k)
                R_WB = self._get_midpoint(self.variables.R_WBs, k)
            else:
                f_c_B = self.variables.f_c_Bs[k]
                p_c_B = self.variables.p_c_Bs[k]
                R_WB = self.variables.R_WBs[k]

            x_dot, dyn = self.quasi_static_dynamics(
                v_WB,
                omega_WB,
                f_c_B,
                p_c_B,
                R_WB,
                force_max,
                torque_max,
            )
            quasi_static_dynamic_constraint = eq(x_dot - dyn, 0)
            for row in quasi_static_dynamic_constraint:
                self.prog.AddConstraint(row)

        # # Angular velocity constraints
        # to_skew_symmetric: Callable[
        #     [float], npt.NDArray[np.float64]
        # ] = lambda omega: np.array([[0.0, -omega], [omega, 0.0]])
        # for omega_WB, R_WB, R_WB_dot in zip(
        #     self.variables.omega_WBs, self.variables.R_WBs, self.variables.R_WB_dots
        # ):
        #     rhs = R_WB_dot.dot(R_WB.T)
        #     lhs = to_skew_symmetric(omega_WB)
        #
        #     constraint = eq(lhs, rhs)
        #     c = constraint[1, 0]
        #     self.prog.AddConstraint(c)
        #     # c = constraint[0, 0]
        #     # self.prog.AddConstraint(c)
        #     # TODO(bernhardpg): Why does constraint[0,0] make the relaxation infeasible?
        #     # This should be a valid constraint
        #
        #     rhs = R_WB_dot
        #     lhs = to_skew_symmetric(omega_WB).dot(R_WB)
        #
        #     constraint = eq(lhs, rhs)
        #     c = constraint[1, 0]
        #     self.prog.AddConstraint(c)
        #     # c = constraint[0, 0]
        #     # self.prog.AddConstraint(c)
        #     # TODO(bernhardpg): Why does constraint[0,0] make the relaxation infeasible?
        #     # This should be a valid constraint

        # Angular velocity constraints
        to_skew_symmetric: Callable[
            [float], npt.NDArray[np.float64]
        ] = lambda omega: np.array([[0.0, -omega], [omega, 0.0]])

        for k in range(self.num_knot_points - 1):
            R_WB_dot = self.variables.R_WB_dots[k]
            omega_WB = self.variables.omega_WBs[k]

            # NOTE: We enforce dynamics at midway points as this is where the velocity is 'valid'
            if use_midpoint:
                R_WB = self._get_midpoint(self.variables.R_WBs, k)
            else:
                R_WB = self.variables.R_WBs[k]

            rhs = R_WB_dot.dot(R_WB.T)
            lhs = to_skew_symmetric(omega_WB)
            constraint = eq(lhs, rhs)
            c = constraint[1, 0]
            self.prog.AddConstraint(c)
            c = constraint[0, 0]
            self.prog.AddConstraint(c)

            rhs = R_WB_dot
            lhs = to_skew_symmetric(omega_WB).dot(R_WB)
            constraint = eq(lhs, rhs)
            c = constraint[1, 0]
            self.prog.AddConstraint(c)
            c = constraint[0, 0]
            self.prog.AddConstraint(c)

        # Ensure sticking on the contact point
        for v_c_B in self.variables.v_c_Bs:
            # NOTE: This is not constraining the real velocity, but it does ensure sticking
            self.prog.AddLinearConstraint(eq(v_c_B, 0))

    def _define_costs(self) -> None:
        # Minimize kinetic energy through squared velocities
        sq_linear_vels = sum([v_WB.T.dot(v_WB) for v_WB in self.variables.v_WBs]).item()  # type: ignore
        self.prog.AddQuadraticCost(self.cost_param_lin_vels * sq_linear_vels)

        sq_angular_vels = np.sum(
            [
                cos_dot**2 + sin_dot**2
                for cos_dot, sin_dot in zip(
                    self.variables.cos_th_dots, self.variables.sin_th_dots
                )
            ]
        )
        self.prog.AddQuadraticCost(self.cost_param_ang_vels * sq_angular_vels)  # type: ignore

        self.prog.AddQuadraticCost(np.sum(self.variables.omega_WBs))  # type: ignore

    def set_finger_pos(self, lam: float) -> None:
        """
        Set finger position along the contact face.
        As the finger position is constant, there is no difference between
        initial and target value.

        @param lam: Position along face, value 0 to 1.
        """
        if lam >= 1 or lam <= 0:
            raise ValueError("The finger position should be set between 0 and 1")

        self.prog.AddLinearConstraint(self.variables.lams[0] == lam)

    def set_slider_initial_pose(self, pose: PlanarPose) -> None:
        self.prog.AddLinearConstraint(self.variables.cos_ths[0] == np.cos(pose.theta))
        self.prog.AddLinearConstraint(self.variables.sin_ths[0] == np.sin(pose.theta))
        self.prog.AddLinearConstraint(eq(self.variables.p_WBs[0], pose.pos()))
        breakpoint()

        self.slider_initial_pose = pose

    def set_slider_final_pose(self, pose: PlanarPose) -> None:
        self.prog.AddLinearConstraint(self.variables.cos_ths[-1] == np.cos(pose.theta))
        self.prog.AddLinearConstraint(self.variables.sin_ths[-1] == np.sin(pose.theta))
        self.prog.AddLinearConstraint(eq(self.variables.p_WBs[-1], pose.pos()))

        self.slider_final_pose = pose

    def formulate_convex_relaxation(self) -> None:
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
        p_WB_xs = self._get_vars_solution_for_vertex_vars(vertex.x(), self.variables.p_WB_xs, result)  # type: ignore
        p_WB_ys = self._get_vars_solution_for_vertex_vars(vertex.x(), self.variables.p_WB_ys, result)  # type: ignore

        omega_WBs = self._get_vars_solution_for_vertex_vars(vertex.x(), self.variables.omega_WBs, result)  # type: ignore

        return FaceContactVariables(
            self.variables.num_knot_points,
            self.variables.time_in_mode,
            self.variables.dt,
            lams,
            normal_forces,
            friction_forces,
            cos_ths,
            sin_ths,
            p_WB_xs,
            p_WB_ys,
            omega_WBs,
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
        p_WB_xs = result.GetSolution(self.variables.p_WB_xs)  # type: ignore
        p_WB_ys = result.GetSolution(self.variables.p_WB_ys)  # type: ignore
        omega_WBs = result.GetSolution(self.variables.omega_WBs)  # type: ignore

        return FaceContactVariables(
            self.variables.num_knot_points,
            self.variables.time_in_mode,
            self.variables.dt,
            lams,
            normal_forces,
            friction_forces,
            cos_ths,
            sin_ths,
            p_WB_xs,
            p_WB_ys,
            omega_WBs,
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
                self.variables.p_c_Bs[0],
                self.variables.p_WBs[0],
                self.variables.cos_ths[0],
                self.variables.sin_ths[0],
            )
        else:
            return ContinuityVariables(
                self.variables.p_c_Bs[-1],
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
        vars = vertex.x()[var_idxs]
        bindings = [Binding[LinearCost](e, v) for e, v in zip(evaluators, vars)]
        for b in bindings:
            vertex.AddCost(b)

    @staticmethod
    def _get_midpoint(vals, k: int):
        return (vals[k] + vals[k + 1]) / 2

    @staticmethod
    def quasi_static_dynamics(
        v_WB,
        omega_WB,
        f_c_B,
        p_c_B,
        R_WB,
        f_max: float,
        tau_max: float,
        use_redundant_constraints: bool = True,
    ):
        A = np.diag(
            [1 / f_max**2, 1 / f_max**2, 1 / tau_max**2]
        )  # Ellipsoidal Limit surface approximation

        # We need to add an entry for multiplication with the wrench,
        # see paper "Reactive Planar Manipulation with Convex Hybrid MPC"
        R = np.zeros((3, 3), dtype="O")
        R[2, 2] = 1
        R[0:2, 0:2] = R_WB

        # Contact torques
        tau_c_B = cross_2d(p_c_B, f_c_B)

        x_dot_in_W = np.concatenate((v_WB, [[omega_WB]]))
        wrench_B = np.concatenate((f_c_B, [[tau_c_B]]))
        wrench_W = R.dot(wrench_B)
        dynamics_in_W = A.dot(
            wrench_W
        )  # Note: A and R are switched here compared to original paper, but A is diagonal so it makes no difference
        if use_redundant_constraints:
            # NOTE(bernhardpg): Add constraints both ways
            x_dot_in_B = R.T.dot(x_dot_in_W)
            dynamics_in_B = A.dot(
                wrench_B
            )  # Note: A and R are switched here compared to original paper, but A is diagonal so it makes no difference

            # x_dot, f(x,u)
            return np.vstack((x_dot_in_W, x_dot_in_B)), np.vstack(
                (dynamics_in_W, dynamics_in_B)
            )  # (6,1), (6,1)
        else:
            return x_dot_in_W, dynamics_in_W  # (3,1), (3,1)
