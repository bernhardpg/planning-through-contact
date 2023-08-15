from dataclasses import dataclass
from typing import Callable, List, Literal, Tuple

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
from planning_through_contact.planning.planar.planar_plan_specs import PlanarPlanSpecs
from planning_through_contact.tools.types import NpExpressionArray, NpVariableArray
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
            self.num_knot_points,
            self.time_in_mode,
            self.dt,
            get_float_from_result(self.lams),
            get_float_from_result(self.normal_forces),
            get_float_from_result(self.friction_forces),
            get_float_from_result(self.cos_ths),
            get_float_from_result(self.sin_ths),
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
            self.num_knot_points,
            self.time_in_mode,
            self.dt,
            get_original_vars_from_reduced(self.lams),
            get_original_vars_from_reduced(self.normal_forces),
            get_original_vars_from_reduced(self.friction_forces),
            get_original_vars_from_reduced(self.cos_ths),
            get_original_vars_from_reduced(self.sin_ths),
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

    @property
    def omega_WBs(self):
        R_WB_dots = [
            np.array([[cos_dot, -sin_dot], [sin_dot, cos_dot]])
            for cos_dot, sin_dot in zip(self.cos_th_dots, self.sin_th_dots)
        ]
        # In 2D, omega_z = theta_dot will be at position (1,0) in R_dot * R'
        oms = [R_dot.dot(R.T)[1, 0] for R, R_dot in zip(self.R_WBs, R_WB_dots)]
        return oms

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
    use_eq_elimination: bool = False
    cost_param_lin_vels: float = 1.0
    cost_param_ang_vels: float = 1.0

    @classmethod
    def create_from_plan_spec(
        cls,
        contact_location: PolytopeContactLocation,
        specs: PlanarPlanSpecs,
        object: RigidBody,
        use_eq_elimination: bool = False,
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
            use_eq_elimination=use_eq_elimination,
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
        self.enforce_equal_forces = True

        self._define_constraints()
        self._define_costs()

    def _define_constraints(self) -> None:
        # TODO: take this from drake simulation
        FRICTION_COEFF = 0.5
        MAX_FORCE = FRICTION_COEFF * self.object.mass * 9.81
        TABLE_SIZE = 1.0

        for lam in self.variables.lams:
            self.prog.AddBoundingBoxConstraint(0, 1, lam)

        # SO(2) constraints
        for c, s in zip(self.variables.cos_ths, self.variables.sin_ths):
            self.prog.AddConstraint(c**2 + s**2 == 1)

        # Friction cone constraints
        for c_n in self.variables.normal_forces:
            self.prog.AddBoundingBoxConstraint(0, MAX_FORCE, c_n)

        for c_n, c_f in zip(
            self.variables.normal_forces, self.variables.friction_forces
        ):
            self.prog.AddLinearConstraint(c_f <= FRICTION_COEFF * c_n)
            self.prog.AddLinearConstraint(c_f >= -FRICTION_COEFF * c_n)

        # Bounds on forces
        for c_n, c_f in zip(
            self.variables.normal_forces, self.variables.friction_forces
        ):
            self.prog.AddBoundingBoxConstraint(-MAX_FORCE, MAX_FORCE, c_f)

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
                v_WB, omega_WB, f_c_B, p_c_B, R_WB, FRICTION_COEFF, self.object.mass
            )
            quasi_static_dynamic_constraint = eq(x_dot - dyn, 0)
            for row in quasi_static_dynamic_constraint:
                self.prog.AddConstraint(row)

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

        self.slider_initial_pose = pose

    def set_slider_final_pose(self, pose: PlanarPose) -> None:
        self.prog.AddLinearConstraint(self.variables.cos_ths[-1] == np.cos(pose.theta))
        self.prog.AddLinearConstraint(self.variables.sin_ths[-1] == np.sin(pose.theta))
        self.prog.AddLinearConstraint(eq(self.variables.p_WBs[-1], pose.pos()))

        self.slider_final_pose = pose

    def formulate_convex_relaxation(self) -> None:
        if self.use_eq_elimination:
            (
                self.reduced_prog,
                self.get_original_vars_from_reduced,
            ) = eliminate_equality_constraints(self.prog)
            self.reduced_variables = self.variables.from_reduced_prog(
                self.prog, self.reduced_prog, self.get_original_vars_from_reduced
            )
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
        p_WB_xs = self._get_vars_solution_for_vertex_vars(vertex.x(), self.variables.p_WB_xs, result)  # type: ignore
        p_WB_ys = self._get_vars_solution_for_vertex_vars(vertex.x(), self.variables.p_WB_ys, result)  # type: ignore

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
        FRICTION_COEFF,
        OBJECT_MASS,
        use_redundant_constraints: bool = True,
    ):
        G = 9.81
        # TODO(bernhardpg): Compute f_max and tau_max correctly
        f_max = FRICTION_COEFF * G * OBJECT_MASS
        tau_max = f_max * 0.2

        A = np.diag(
            # [1 / f_max**2, 1 / f_max**2, 1 / tau_max**2]
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
