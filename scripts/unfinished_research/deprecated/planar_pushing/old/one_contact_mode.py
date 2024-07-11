import argparse
import time
from typing import List, NamedTuple, Optional, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pydot
import pydrake.geometry.optimization as opt
import pydrake.symbolic as sym
from pydrake.math import eq, ge, le
from pydrake.solvers import (
    CommonSolverOption,
    IpoptSolver,
    MakeSemidefiniteRelaxation,
    MathematicalProgram,
    MathematicalProgramResult,
    MixedIntegerBranchAndBound,
    MosekSolver,
    SnoptSolver,
    Solve,
    SolverOptions,
)
from pydrake.trajectories import PiecewisePolynomial, PiecewiseQuaternionSlerp

from planning_through_contact.convex_relaxation.sdp import (
    create_sdp_relaxation,
    eliminate_equality_constraints,
)
from planning_through_contact.geometry.polyhedron import PolyhedronFormulator
from planning_through_contact.geometry.two_d.contact.types import ContactLocation
from planning_through_contact.geometry.two_d.equilateral_polytope_2d import (
    EquilateralPolytope2d,
)
from planning_through_contact.geometry.two_d.rigid_body_2d import (
    PolytopeContactLocation,
    RigidBody2d,
)
from planning_through_contact.geometry.two_d.t_pusher import TPusher
from planning_through_contact.geometry.utilities import cross_2d
from planning_through_contact.tools.types import (
    NpExpressionArray,
    NpFormulaArray,
    NpVariableArray,
)
from planning_through_contact.visualize.analysis import (
    create_quasistatic_pushing_analysis,
    plot_cos_sine_trajs,
)
from planning_through_contact.visualize.colors import COLORS
from planning_through_contact.visualize.visualizer_2d import (
    VisualizationForce2d,
    VisualizationPoint2d,
    VisualizationPolygon2d,
    Visualizer2d,
)

T = TypeVar("T")


def face_name(face_idx: float) -> str:
    return f"face_{face_idx}"


def non_collision_name(face_idx: float) -> str:
    return f"non_collision_{face_idx}"


def forward_differences(vars, dt: float):
    # TODO: It is cleaner to implement this using a forward diff matrix, but as a first step I do this the simplest way
    forward_diffs = [
        (var_next - var_curr) / dt for var_curr, var_next in zip(vars[0:-1], vars[1:])  # type: ignore
    ]
    return forward_diffs


def make_so3(R: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Takes a SO(2) rotation matrix and returns a rotation matrix in SO(3), where the original matrix
    is treated as a rotation about the z-axis.
    """
    R_in_SO3 = np.eye(3)
    R_in_SO3[0:2, 0:2] = R
    return R_in_SO3


def interpolate_so2_using_slerp(
    Rs: List[npt.NDArray[np.float64]],
    start_time: float,
    end_time: float,
    dt: float,
) -> List[npt.NDArray[np.float64]]:
    """
    Assumes evenly spaced knot points R_matrices.

    @return: trajectory evaluated evenly at every dt-th step, starting at start_time and ending at specified end_time.
    """

    Rs_in_SO3 = [make_so3(R) for R in Rs]
    knot_point_times = np.linspace(start_time, end_time, len(Rs))
    quat_slerp_traj = PiecewiseQuaternionSlerp(knot_point_times, Rs_in_SO3)

    traj_times = np.arange(start_time, end_time, dt)
    R_traj_in_SO2 = [
        quat_slerp_traj.orientation(t).rotation()[0:2, 0:2] for t in traj_times
    ]

    return R_traj_in_SO2


def interpolate_w_first_order_hold(
    values: npt.NDArray[np.float64],  # (NUM_SAMPLES, NUM_DIMS)
    start_time: float,
    end_time: float,
    dt: float,
) -> npt.NDArray[np.float64]:  # (NUM_POINTS, NUM_DIMS)
    """
    Assumes evenly spaced knot points.

    @return: trajectory evaluated evenly at every dt-th step, starting at start_time and ending at specified end_time.
    """

    knot_point_times = np.linspace(start_time, end_time, len(values))

    # Drake expects the values to be (NUM_DIMS, NUM_SAMPLES)
    first_order_hold = PiecewisePolynomial.FirstOrderHold(knot_point_times, values.T)
    traj_times = np.arange(start_time, end_time, dt)
    traj = np.hstack(
        [first_order_hold.value(t) for t in traj_times]
    ).T  # transpose to match format in rest of project

    return traj


# TODO: should probably have a better name
class ModeVars(NamedTuple):
    # NOTE: These types are wrong
    lams: NpVariableArray  # (1, num_knot_points)
    normal_forces: NpVariableArray  # (1, num_knot_points)
    friction_forces: NpVariableArray  # (1, num_knot_points)
    cos_ths: NpVariableArray  # (1, num_knot_points)
    sin_ths: NpVariableArray  # (1, num_knot_points)
    p_WB_xs: NpVariableArray  # (1, num_knot_points)
    p_WB_ys: NpVariableArray  # (1, num_knot_points)

    time_in_mode: float

    pv1: npt.NDArray[np.float64]
    pv2: npt.NDArray[np.float64]
    normal_vec: npt.NDArray[np.float64]
    tangent_vec: npt.NDArray[np.float64]
    dt: float

    @classmethod
    def make(
        cls,
        prog: MathematicalProgram,
        object: RigidBody2d,
        contact_face: PolytopeContactLocation,
        num_knot_points: int,
        time_in_mode: float,
    ) -> "ModeVars":
        # Contact positions
        lams = prog.NewContinuousVariables(num_knot_points, "lam")
        pv1, pv2 = object.get_proximate_vertices_from_location(contact_face)

        # Contact forces
        normal_forces = prog.NewContinuousVariables(num_knot_points, "c_n")
        friction_forces = prog.NewContinuousVariables(num_knot_points, "c_f")
        (
            normal_vec,
            tangent_vec,
        ) = object.get_norm_and_tang_vecs_from_location(contact_face)

        # Rotations
        cos_ths = prog.NewContinuousVariables(num_knot_points, "cos_th")
        sin_ths = prog.NewContinuousVariables(num_knot_points, "sin_th")

        # Box position relative to world frame
        p_WB_xs = prog.NewContinuousVariables(num_knot_points, "p_WB_x")
        p_WB_ys = prog.NewContinuousVariables(num_knot_points, "p_WB_y")

        dt = time_in_mode / num_knot_points

        return cls(
            lams,
            normal_forces,
            friction_forces,
            cos_ths,
            sin_ths,
            p_WB_xs,
            p_WB_ys,
            time_in_mode,
            pv1,
            pv2,
            normal_vec,
            tangent_vec,
            dt,
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
    def omega_WBs(self):
        R_WB_dots = [
            np.array([[cos_dot, -sin_dot], [sin_dot, cos_dot]])
            for cos_dot, sin_dot in zip(self.cos_th_dots, self.sin_th_dots)
        ]
        # In 2D, omega_z = theta_dot will be at position (1,0) in R_dot * R'
        oms = [R_dot.dot(R.T)[1, 0] for R, R_dot in zip(self.R_WBs, R_WB_dots)]
        return oms

    def eval_from_vec(
        self, x: npt.NDArray[np.float64], prog: MathematicalProgram
    ) -> "ModeVars":
        """
        Needs the prog to map variables through correct indices
        """

        lam_vals = x[prog.FindDecisionVariableIndices(self.lams)]
        cos_th_vals = x[prog.FindDecisionVariableIndices(self.cos_ths)]
        sin_th_vals = x[prog.FindDecisionVariableIndices(self.sin_ths)]
        normal_force_vals = x[prog.FindDecisionVariableIndices(self.normal_forces)]
        friction_force_vals = x[prog.FindDecisionVariableIndices(self.friction_forces)]
        p_WB_x_vals = x[prog.FindDecisionVariableIndices(self.p_WB_xs)]
        p_WB_y_vals = x[prog.FindDecisionVariableIndices(self.p_WB_ys)]

        return ModeVars(
            lam_vals,
            normal_force_vals,
            friction_force_vals,
            cos_th_vals,
            sin_th_vals,
            p_WB_x_vals,
            p_WB_y_vals,
            self.time_in_mode,
            self.pv1,
            self.pv2,
            self.normal_vec,
            self.tangent_vec,
            self.dt,
        )

    def eval_result(self, result: MathematicalProgramResult) -> "ModeVars":
        lam_vals = result.GetSolution(self.lams)
        cos_th_vals = result.GetSolution(self.cos_ths)
        sin_th_vals = result.GetSolution(self.sin_ths)
        normal_force_vals = result.GetSolution(self.normal_forces)
        friction_force_vals = result.GetSolution(self.friction_forces)
        p_WB_x_vals = result.GetSolution(self.p_WB_xs)
        p_WB_y_vals = result.GetSolution(self.p_WB_ys)

        return ModeVars(
            lam_vals,
            normal_force_vals,
            friction_force_vals,
            cos_th_vals,
            sin_th_vals,
            p_WB_x_vals,
            p_WB_y_vals,
            self.time_in_mode,
            self.pv1,
            self.pv2,
            self.normal_vec,
            self.tangent_vec,
            self.dt,
        )

    @property
    def p_c_Ws(self) -> List[npt.NDArray[np.float64]]:
        return [
            p_WB + R_WB.dot(p_c_B)
            for p_WB, R_WB, p_c_B in zip(self.p_WBs, self.R_WBs, self.p_c_Bs)
        ]

    @property
    def f_c_Ws(self) -> List[npt.NDArray[np.float64]]:
        return [R_WB.dot(f_c_B) for f_c_B, R_WB in zip(self.f_c_Bs, self.R_WBs)]

    # Need to handle R_traj as a special case due to List[(2x2)] structure
    def get_R_traj(
        self, dt: float, interpolate: bool = False
    ) -> List[npt.NDArray[np.float64]]:
        if interpolate:
            return interpolate_so2_using_slerp(self.R_WBs, 0, self.time_in_mode, dt)
        else:
            return self.R_WBs

    def _get_traj(
        self,
        point_sequence: List[npt.NDArray[np.float64]],
        dt: float,
        interpolate: bool = False,
    ) -> npt.NDArray[np.float64]:  # (N, 2)
        knot_points = np.hstack(point_sequence)  # (2, num_knot_points)
        if interpolate:
            return interpolate_w_first_order_hold(
                knot_points.T, 0, self.time_in_mode, dt
            )
        else:
            return knot_points.T

    def get_p_WB_traj(
        self, dt: float, interpolate: bool = False
    ) -> npt.NDArray[np.float64]:
        return self._get_traj(self.p_WBs, dt, interpolate)

    def get_p_c_W_traj(
        self, dt: float, interpolate: bool = False
    ) -> npt.NDArray[np.float64]:
        return self._get_traj(self.p_c_Ws, dt, interpolate)

    def get_f_c_W_traj(
        self, dt: float, interpolate: bool = False
    ) -> npt.NDArray[np.float64]:
        return self._get_traj(self.f_c_Ws, dt, interpolate)

    def get_v_WB_traj(
        self, dt: float, interpolate: bool = False
    ) -> npt.NDArray[np.float64]:
        # Pad with zero to avoid wrong length (velocities have one less element due to finite diffs)
        num_dims = 2
        return self._get_traj(self.v_WBs + [np.zeros((num_dims, 1))], dt, interpolate)  # type: ignore

    def get_omega_WB_traj(
        self, dt: float, interpolate: bool = False
    ) -> npt.NDArray[np.float64]:
        # Pad with zero to avoid wrong length (velocities have one less element due to finite diffs)
        return self._get_traj(self.omega_WBs + [0], dt, interpolate)  # type: ignore


def quasi_static_dynamics(
    v_WB, omega_WB, f_c_B, p_c_B, R_WB, FRICTION_COEFF, OBJECT_MASS
):
    G = 9.81
    f_max = FRICTION_COEFF * G * OBJECT_MASS
    tau_max = f_max * 0.2  # TODO: change this!

    A = np.diag(
        [1 / f_max**2, 1 / f_max**2, 1 / tau_max**2]
    )  # Ellipsoidal Limit surface approximation

    # We need to add an entry for multiplication with the wrench, see paper "Reactive Planar Manipulation with Convex Hybrid MPC"
    R = np.zeros((3, 3), dtype="O")
    R[2, 2] = 1
    R[0:2, 0:2] = R_WB

    # Contact torques
    tau_c_B = cross_2d(p_c_B, f_c_B)

    x_dot = np.concatenate((v_WB, [[omega_WB]]))
    wrench_B = np.concatenate((f_c_B, [[tau_c_B]]))
    wrench_W = R.dot(wrench_B)
    dynamics = A.dot(
        wrench_W
    )  # Note: A and R are switched here compared to original paper, but A is diagonal so it makes no difference

    return x_dot, dynamics  # x_dot, f(x,u)


class PlanarPushingContactMode:
    def __init__(
        self,
        object: RigidBody2d,
        contact_face_idx: int,
        num_knot_points: int = 4,
        end_time: float = 3,
        th_initial: Optional[float] = None,
        th_target: Optional[float] = None,
        pos_initial: Optional[npt.NDArray[np.float64]] = None,
        pos_target: Optional[npt.NDArray[np.float64]] = None,
    ):
        self.name = face_name(contact_face_idx)

        self.num_knot_points = num_knot_points
        self.object = object
        self.time_in_mode = end_time

        self.FRICTION_COEFF = 0.5

        dt = end_time / num_knot_points
        self.dt = dt

        self.prog = MathematicalProgram()

        contact_face = PolytopeContactLocation(
            pos=ContactLocation.FACE, idx=contact_face_idx
        )

        self.vars = ModeVars.make(
            self.prog, object, contact_face, num_knot_points, end_time
        )

        for lam in self.vars.lams:
            self.prog.AddLinearConstraint(lam >= 0)
            self.prog.AddLinearConstraint(lam <= 1)

        # SO(2) constraints
        for c, s in zip(self.vars.cos_ths, self.vars.sin_ths):
            self.prog.AddQuadraticConstraint(c**2 + s**2 - 1, 0, 0)

        # Friction cone constraints
        for c_n in self.vars.normal_forces:
            self.prog.AddLinearConstraint(c_n >= 0)
        for c_n, c_f in zip(self.vars.normal_forces, self.vars.friction_forces):
            self.prog.AddLinearConstraint(c_f <= self.FRICTION_COEFF * c_n)
            self.prog.AddLinearConstraint(c_f >= -self.FRICTION_COEFF * c_n)

        # Quasi-static dynamics
        for k in range(num_knot_points - 1):
            v_WB = self.vars.v_WBs[k]
            omega_WB = self.vars.omega_WBs[k]

            # NOTE: We enforce dynamics at midway points as this is where the velocity is 'valid'
            f_c_B = (self.vars.f_c_Bs[k] + self.vars.f_c_Bs[k + 1]) / 2
            p_c_B = (self.vars.p_c_Bs[k] + self.vars.p_c_Bs[k + 1]) / 2
            R_WB = (self.vars.R_WBs[k] + self.vars.R_WBs[k + 1]) / 2

            # f_c_B = self.vars.f_c_Bs[k]
            # p_c_B = self.vars.p_c_Bs[k]
            # R_WB = self.vars.R_WBs[k]

            x_dot, dyn = quasi_static_dynamics(
                v_WB, omega_WB, f_c_B, p_c_B, R_WB, self.FRICTION_COEFF, object.mass
            )
            quasi_static_dynamic_constraint = x_dot - dyn
            for row in quasi_static_dynamic_constraint.flatten():
                self.prog.AddQuadraticConstraint(row, 0, 0)

        # Ensure sticking on the contact point
        # TODO: remove this
        v_c_Bs = forward_differences(
            self.vars.p_c_Bs, dt
        )  # NOTE: Not real velocity, only time differentiation of coordinates (not equal as B is not an inertial frame)!
        for v_c_B in v_c_Bs:
            self.prog.AddLinearConstraint(
                eq(v_c_B, 0)
            )  # no velocity on contact points in body frame

        # Minimize kinetic energy through squared velocities
        sq_linear_vels = sum([v_WB.T.dot(v_WB) for v_WB in self.vars.v_WBs]).item()  # type: ignore
        sq_angular_vels = sum(
            [
                cos_dot**2 + sin_dot**2
                for cos_dot, sin_dot in zip(
                    self.vars.cos_th_dots, self.vars.sin_th_dots
                )
            ]
        )
        self.prog.AddQuadraticCost(sq_linear_vels)
        self.prog.AddQuadraticCost(sq_angular_vels)

        def create_R(th: float) -> npt.NDArray[np.float64]:
            return np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])

        # Initial conditions (only first and last vertex will have this)
        if th_initial is not None:
            R_WB_I = create_R(th_initial)
            self.prog.AddLinearConstraint(eq(self.vars.R_WBs[0], R_WB_I))
        if th_target is not None:
            R_WB_T = create_R(th_target)
            self.prog.AddLinearConstraint(eq(self.vars.R_WBs[-1], R_WB_T))
        if pos_initial is not None:
            assert pos_initial.shape == (2, 1)
            self.prog.AddLinearConstraint(eq(self.vars.p_WBs[0], pos_initial))
        if pos_target is not None:
            assert pos_target.shape == (2, 1)
            self.prog.AddLinearConstraint(eq(self.vars.p_WBs[-1], pos_target))

        start = time.time()

        print("Starting to create SDP relaxation...")
        self.relaxed_prog = MakeSemidefiniteRelaxation(self.prog)
        end = time.time()
        print(
            f"Finished formulating relaxed problem. Elapsed time: {end - start} seconds"
        )


def plan_planar_pushing(
    experiment_number: int, compute_violation: bool, round_solution: bool
):
    if experiment_number == 0:
        th_initial = 0
        th_target = 0.1
        pos_initial = np.array([[0.0, 0.5]]).T
        pos_target = np.array([[-0.3, 0.2]]).T
        contact_face_idx = 0
    elif experiment_number == 1:
        th_initial = -np.pi / 4
        th_target = -np.pi / 4 + 0.1
        pos_initial = np.array([[-0.3, 0.5]]).T
        pos_target = np.array([[-0.2, 0.2]]).T
        contact_face_idx = 3
    elif experiment_number == 2:
        th_initial = 0
        th_target = 1.0
        pos_initial = np.array([[-0.3, 0.5]]).T
        pos_target = np.array([[0.2, 0.2]]).T
        contact_face_idx = 3
    else:
        th_initial = 0
        th_target = 0.8
        pos_initial = np.array([[0.0, 0.5]]).T
        pos_target = np.array([[-0.3, 0.2]]).T
        contact_face_idx = 0

    num_knot_points = 8
    time_in_contact = 2

    MASS = 0.3
    DIST_TO_CORNERS = 0.2
    VIS_REALTIME_RATE = 0.25
    num_vertices = 4

    object = EquilateralPolytope2d(
        actuated=False,
        name="Slider",
        mass=MASS,
        vertex_distance=DIST_TO_CORNERS,
        num_vertices=num_vertices,
    )

    contact_mode = PlanarPushingContactMode(
        object,
        num_knot_points=num_knot_points,
        contact_face_idx=contact_face_idx,
        end_time=time_in_contact,
        th_initial=th_initial,
        pos_initial=pos_initial,
        th_target=th_target,
        pos_target=pos_target,
    )

    # Solve the problem by using elimination of equality constraints
    eliminate_equalities = True

    NUM_TRIALS = 5
    DEBUG = False

    solver_options = SolverOptions()
    if DEBUG:
        solver_options.SetOption(CommonSolverOption.kPrintToConsole, 1)  # type: ignore

    elapsed_times = []
    for _ in range(NUM_TRIALS):
        if eliminate_equalities:
            smaller_prog, retrieve_x = eliminate_equality_constraints(
                contact_mode.prog, print_num_vars_eliminated=True
            )
            # TODO(bernhardpg): This is consistently slower to solve than my implementation
            relaxed_prog = MakeSemidefiniteRelaxation(smaller_prog)
            # relaxed_prog, _, _ = create_sdp_relaxation(smaller_prog)
        else:
            relaxed_prog = contact_mode.relaxed_prog

        start_time = time.time()
        relaxed_result = Solve(relaxed_prog, solver_options=solver_options)
        end_time = time.time()
        assert relaxed_result.is_success()

        elapsed_time = end_time - start_time
        elapsed_times.append(elapsed_time)

    print(f"Mean elapsed time: {np.mean(elapsed_times)}")

    if eliminate_equalities:
        z_sols = relaxed_result.GetSolution(smaller_prog.decision_variables())  # type: ignore
        decision_var_vals = sym.Evaluate(retrieve_x(z_sols)).flatten()  # type: ignore
        relaxed_sols = contact_mode.vars.eval_from_vec(decision_var_vals, contact_mode.prog)  # type: ignore

    else:
        decision_var_vals = relaxed_result.GetSolution(
            contact_mode.prog.decision_variables()
        )
        relaxed_sols = contact_mode.vars.eval_result(relaxed_result)  # type: ignore

    vals = [
        relaxed_sols
    ]  # Will have one val per contact mode, but here there's only one

    if round_solution:
        print("Solving nonlinear trajopt...")

        solver_options = SolverOptions()
        solver_options.SetOption(CommonSolverOption.kPrintToConsole, 1)  # type: ignore
        solver_options.SetOption(IpoptSolver.id(), "tol", 1e-6)

        contact_mode.prog.SetInitialGuess(
            contact_mode.prog.decision_variables(), decision_var_vals
        )

        snopt = SnoptSolver()
        true_result = snopt.Solve(contact_mode.prog, solver_options=solver_options)  # type: ignore
        assert true_result.is_success()
        print("Found solution to true problem!")

    DT = 0.01
    interpolate = False
    R_traj = sum(
        [val.get_R_traj(DT, interpolate=interpolate) for val in vals],
        [],
    )

    # Make sure rotation relaxation is tight
    for R in R_traj:
        det = np.abs(np.linalg.det(R))
        eps = 0.01
        assert det <= 1 + eps and det >= 1 - eps

    com_traj = np.vstack(
        [val.get_p_WB_traj(DT, interpolate=interpolate) for val in vals]
    )
    force_traj = np.vstack(
        [val.get_f_c_W_traj(DT, interpolate=interpolate) for val in vals]
    )
    contact_pos_traj = np.vstack(
        [val.get_p_c_W_traj(DT, interpolate=interpolate) for val in vals]
    )
    object_vel_traj = np.vstack(
        [val.get_v_WB_traj(DT, interpolate=interpolate) for val in vals]
    )
    object_ang_vel_traj = np.vstack(
        [val.get_omega_WB_traj(DT, interpolate=interpolate) for val in vals]
    )
    traj_length = len(R_traj)

    # Print mean velocities for error plots
    max_trans_vel = np.max([np.linalg.norm(vel) for vel in object_vel_traj])
    max_ang_vel = np.max(object_ang_vel_traj)
    print(f"Max translational velocity: {max_trans_vel}")
    print(f"Max angular velocity: {max_ang_vel}")

    vars = vals[0]
    if compute_violation:
        quasi_static_violation = []
        for k in range(traj_length - 1):
            v_WB = vars.v_WBs[k]
            omega_WB = vars.omega_WBs[k]

            # NOTE: We enforce dynamics at midway points as this is where the velocity is 'valid'
            f_c_B = (vars.f_c_Bs[k] + vars.f_c_Bs[k + 1]) / 2
            p_c_B = (vars.p_c_Bs[k] + vars.p_c_Bs[k + 1]) / 2
            R_WB = (vars.R_WBs[k] + vars.R_WBs[k + 1]) / 2

            # f_c_B = vars.f_c_Bs[k]
            # p_c_B = vars.p_c_Bs[k]
            # R_WB = vars.R_WBs[k]

            x_dot, dyn = quasi_static_dynamics(
                v_WB,
                omega_WB,
                f_c_B,
                p_c_B,
                R_WB,
                contact_mode.FRICTION_COEFF,
                object.mass,
            )
            violation = x_dot - dyn
            quasi_static_violation.append(violation)

        quasi_static_violation = np.hstack(
            quasi_static_violation
        ).T  # (N, num_knot_points)
        create_quasistatic_pushing_analysis(
            quasi_static_violation,
            num_knot_points,
            trans_velocity_ref=max_trans_vel,
            angular_velocity_ref=max_ang_vel,
        )
        plt.show()

    CONTACT_COLOR = COLORS["dodgerblue4"]
    GRAVITY_COLOR = COLORS["blueviolet"]
    BOX_COLOR = COLORS["aquamarine4"]
    TABLE_COLOR = COLORS["bisque3"]
    FINGER_COLOR = COLORS["firebrick3"]
    TARGET_COLOR = COLORS["firebrick1"]
    VELOCITY_COLOR = COLORS["darkorange1"]

    flattened_rotation = np.vstack([R.flatten() for R in R_traj])
    box_viz = VisualizationPolygon2d.from_trajs(
        com_traj,
        flattened_rotation,
        object,
        BOX_COLOR,
    )

    # NOTE: I don't really need the entire trajectory here, but leave for now
    target_viz = VisualizationPolygon2d.from_trajs(
        com_traj,
        flattened_rotation,
        object,
        TARGET_COLOR,
    )

    com_points_viz = VisualizationPoint2d(com_traj, GRAVITY_COLOR)  # type: ignore
    contact_point_viz = VisualizationPoint2d(contact_pos_traj, FINGER_COLOR)  # type: ignore
    contact_force_viz = VisualizationForce2d(contact_pos_traj, CONTACT_COLOR, force_traj)  # type: ignore

    # visualize velocity with an arrow (i.e. as a force), and reverse force scaling
    VEL_VIZ_SCALE_CONSTANT = 0.3
    object_vel_viz = VisualizationForce2d(com_traj, VELOCITY_COLOR, object_vel_traj / VEL_VIZ_SCALE_CONSTANT)  # type: ignore

    viz = Visualizer2d()
    FRAMES_PER_SEC = len(R_traj) / (time_in_contact / VIS_REALTIME_RATE)
    viz.visualize(
        [contact_point_viz],
        [contact_force_viz, object_vel_viz],
        [box_viz],
        FRAMES_PER_SEC,
        target_viz,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--experiment",
        help="Which experiment to run",
        type=int,
        default=0,
    )
    parser.add_argument(
        "-c",
        "--compute_violation",
        help="Display relaxation error plot",
        action="store_true",
    )
    parser.add_argument(
        "-r",
        "--round",
        help="Round solution using nonlinear trajopt",
        action="store_true",
    )
    args = parser.parse_args()
    experiment_number = args.experiment
    compute_violation = args.compute_violation
    round_solution = args.round

    plan_planar_pushing(experiment_number, compute_violation, round_solution)
