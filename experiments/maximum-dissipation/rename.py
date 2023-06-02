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
    MathematicalProgram,
    MathematicalProgramResult,
    MixedIntegerBranchAndBound,
    MosekSolver,
    Solve,
    SolverOptions,
)
from pydrake.trajectories import PiecewisePolynomial, PiecewiseQuaternionSlerp

from convex_relaxation.sdp import create_sdp_relaxation
from geometry.polyhedron import PolyhedronFormulator
from geometry.two_d.contact.types import ContactLocation
from geometry.two_d.equilateral_polytope_2d import EquilateralPolytope2d
from geometry.two_d.rigid_body_2d import PolytopeContactLocation, RigidBody2d
from geometry.two_d.t_pusher import TPusher
from geometry.utilities import cross_2d
from tools.types import NpExpressionArray, NpFormulaArray, NpVariableArray
from visualize.analysis import create_quasistatic_pushing_analysis
from visualize.colors import COLORS
from visualize.visualizer_2d import (
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


def forward_differences(
    vars: List[NpExpressionArray], dt: float
) -> List[NpExpressionArray]:
    # TODO: It is cleaner to implement this using a forward diff matrix, but as a first step I do this the simplest way
    forward_diffs = [
        (var_next - var_curr) / dt for var_curr, var_next in zip(vars[0:-1], vars[1:])
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


class ModeVarsResult(NamedTuple):
    cos_ths: npt.NDArray[np.float64]
    sin_ths: npt.NDArray[np.float64]
    p_WBs: npt.NDArray[np.float64]
    p_c_Bs: npt.NDArray[np.float64]
    f_c_Bs: npt.NDArray[np.float64]
    time_in_mode: float

    @property
    def num_knot_points(self) -> int:
        return len(self.R_WBs)

    @property
    def R_WBs(self) -> List[npt.NDArray[np.float64]]:
        Rs = [
            np.array([[cos, -sin], [sin, cos]])
            for cos, sin in zip(self.cos_ths, self.sin_ths)
        ]
        return Rs

    @property
    def v_WBs(self) -> npt.NDArray[np.float64]:
        dt = self.num_knot_points / self.time_in_mode
        forward_diffs = (self.p_WBs[:, 1:] - self.p_WBs[:, 0:-1]) / dt
        # NOTE: quick fix
        res = np.hstack(
            (forward_diffs, np.zeros((2, 1)))
        )  # add zeros to the end to make velocity traj as long as all other trajs
        return res

    def _rotate_to_world(
        self, vecs_B: npt.NDArray[np.float64]  # (2, num_knot_points)
    ) -> npt.NDArray[np.float64]:
        vecs_W = np.hstack(
            [np.expand_dims(R.dot(v), 1) for R, v in zip(self.R_WBs, vecs_B.T)]
        )
        return vecs_W  # (2, num_knot_points)

    @property
    def p_c_Ws(self) -> npt.NDArray[np.float64]:
        return self.p_WBs + self._rotate_to_world(self.p_c_Bs)

    @property
    def f_c_Ws(self) -> npt.NDArray[np.float64]:
        return self.f_c_Bs

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
        knot_points: npt.NDArray[np.float64],
        dt: float,
        interpolate: bool = False,
    ) -> npt.NDArray[np.float64]:
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
        return self._get_traj(self.v_WBs, dt, interpolate)


# TODO: should probably have a better name
class ModeVars(NamedTuple):
    # TODO: Refactor all code so that it just creates a ModeVars object, which instantiates all the variables (based on prog)
    cos_ths: NpVariableArray  # (1, num_knot_points)
    sin_ths: NpVariableArray  # (1, num_knot_points)
    p_WBs: NpVariableArray  # (2, num_knot_points)
    p_c_Bs: NpExpressionArray  # (2, num_knot_points)
    f_c_Bs: NpExpressionArray  # (2, num_knot_points)
    time_in_mode: float

    def eval_result(self, result: MathematicalProgramResult) -> ModeVarsResult:
        cos_th_vals = result.GetSolution(self.cos_ths)
        sin_th_vals = result.GetSolution(self.sin_ths)
        p_WB_vals = result.GetSolution(self.p_WBs)

        # When result.GetSolution() is called on an expression, it returns an expression, not float
        eval_expr_on_vector = np.vectorize(
            lambda x: x.Evaluate() if isinstance(x, sym.Expression) else x
        )

        p_c_B_vals = eval_expr_on_vector(result.GetSolution(self.p_c_Bs))
        f_c_B_vals = eval_expr_on_vector(result.GetSolution(self.f_c_Bs))

        # TODO: Temporary fix to debug through visualization.
        # NOTE: We enforce dynamics at mid-way points, so plot mid-way points
        # (this is also how we compute vels)
        def make_mean(vec):
            means = (vec[:, 0:-1] + vec[:, 1:]) / 2
            padded_means = np.hstack((means, np.zeros((2, 1))))

            return padded_means

        p_c_B_vals = make_mean(p_c_B_vals)
        f_c_B_vals = make_mean(f_c_B_vals)

        return ModeVarsResult(
            cos_th_vals,
            sin_th_vals,
            p_WB_vals,
            p_c_B_vals,
            f_c_B_vals,
            self.time_in_mode,
        )


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

        FRICTION_COEFF = 0.5
        G = 9.81

        f_max = FRICTION_COEFF * G * object.mass
        tau_max = f_max * 0.2  # TODO change this!

        A = np.diag(
            [1 / f_max**2, 1 / f_max**2, 1 / tau_max**2]
        )  # Ellipsoidal Limit surface approximation

        dt = end_time / num_knot_points
        self.dt = dt

        prog = MathematicalProgram()

        contact_face = PolytopeContactLocation(
            pos=ContactLocation.FACE, idx=contact_face_idx
        )

        # Contact positions
        lams = prog.NewContinuousVariables(num_knot_points, "lam")
        for lam in lams:
            prog.AddLinearConstraint(lam >= 0)
            prog.AddLinearConstraint(lam <= 1)

        self.pv1, self.pv2 = self.object.get_proximate_vertices_from_location(
            contact_face
        )
        p_c_Bs = [lam * self.pv1 + (1 - lam) * self.pv2 for lam in lams]

        # Contact forces
        normal_forces = prog.NewContinuousVariables(num_knot_points, "c_n")
        friction_forces = prog.NewContinuousVariables(num_knot_points, "c_f")
        (
            self.normal_vec,
            self.tangent_vec,
        ) = self.object.get_norm_and_tang_vecs_from_location(contact_face)
        f_c_Bs = [
            c_n * self.normal_vec + c_f * self.tangent_vec
            for c_n, c_f in zip(normal_forces, friction_forces)
        ]

        # Rotations
        cos_ths = prog.NewContinuousVariables(num_knot_points, "cos_th")
        sin_ths = prog.NewContinuousVariables(num_knot_points, "sin_th")
        R_WBs = [
            np.array([[cos, sin], [-sin, cos]]) for cos, sin in zip(cos_ths, sin_ths)
        ]

        # Box position relative to world frame
        p_WB_xs = prog.NewContinuousVariables(num_knot_points, "p_WB_x")
        p_WB_ys = prog.NewContinuousVariables(num_knot_points, "p_WB_y")
        p_WBs = [np.array([x, y]) for x, y in zip(p_WB_xs, p_WB_ys)]

        # Compute velocities
        v_WBs = forward_differences(p_WBs, dt)
        cos_th_dots = forward_differences(cos_ths, dt)
        sin_th_dots = forward_differences(sin_ths, dt)
        R_WB_dots = [
            np.array([[cos_dot, -sin_dot], [sin_dot, cos_dot]])
            for cos_dot, sin_dot in zip(cos_th_dots, sin_th_dots)
        ]
        v_c_Bs = forward_differences(
            p_c_Bs, dt
        )  # NOTE: Not real velocity, only time differentiation of coordinates (not equal as B is not an inertial frame)!

        # In 2D, omega_z = theta_dot will be at position (0,1) in R_dot * R'
        omega_WBs = [R_dot.dot(R.T)[1, 0] for R, R_dot in zip(R_WBs, R_WB_dots)]

        # SO(2) constraints
        for c, s in zip(cos_ths, sin_ths):
            prog.AddConstraint(c**2 + s**2 == 1)

        # # Friction cone constraints
        for c_n in normal_forces:
            prog.AddLinearConstraint(c_n >= 0)
        for c_n, c_f in zip(normal_forces, friction_forces):
            prog.AddLinearConstraint(c_f <= FRICTION_COEFF * c_n)
            prog.AddLinearConstraint(c_f >= -FRICTION_COEFF * c_n)

        # Quasi-static dynamics
        for k in range(num_knot_points - 1):
            v_WB = v_WBs[k]
            omega_WB = omega_WBs[k]
            # NOTE: We enforce dynamics at midway points
            f_c_B = (f_c_Bs[k] + f_c_Bs[k + 1]) / 2
            p_c_B = (p_c_Bs[k] + p_c_Bs[k + 1]) / 2
            R_WB = (R_WBs[k] + R_WBs[k + 1]) / 2

            # f_c_B = f_c_Bs[k]
            # p_c_B = p_c_Bs[k]
            # R_WB = R_WBs[k]

            # We need to add an entry for multiplication with the wrench, see paper "Reactive Planar Manipulation with Convex Hybrid MPC"
            R = np.zeros((3, 3), dtype="O")
            R[2, 2] = 1
            R[0:2, 0:2] = R_WB

            # Contact torques
            tau_c_B = cross_2d(p_c_B, f_c_B)

            x_dot = np.concatenate([v_WB, [omega_WB]])
            wrench_B = np.concatenate(
                [f_c_B.flatten(), [tau_c_B]]
            )  # NOTE: Should fix not nice vector dimensions
            wrench_W = R.dot(wrench_B)

            quasi_static_dynamic_constraint = eq(x_dot, A.dot(wrench_W))
            for row in quasi_static_dynamic_constraint:
                prog.AddConstraint(row)

        # Ensure sticking on the contact point
        for v_c_B in v_c_Bs:
            prog.AddLinearConstraint(
                eq(v_c_B, 0)
            )  # no velocity on contact points in body frame

        # Minimize kinetic energy through squared velocities
        sq_linear_vels = sum([v_WB.T.dot(v_WB) for v_WB in v_WBs])
        sq_angular_vels = sum(
            [
                cos_dot**2 + sin_dot**2
                for cos_dot, sin_dot in zip(cos_th_dots, sin_th_dots)
            ]
        )
        prog.AddQuadraticCost(sq_linear_vels)
        prog.AddQuadraticCost(sq_angular_vels)

        def create_R(th: float) -> npt.NDArray[np.float64]:
            return np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])

        # Initial conditions (only first and last vertex will have this)
        if th_initial is not None:
            R_WB_I = create_R(th_initial)
            prog.AddLinearConstraint(eq(R_WBs[0], R_WB_I))
        if th_target is not None:
            R_WB_T = create_R(th_target)
            prog.AddLinearConstraint(eq(R_WBs[-1], R_WB_T))
        if pos_initial is not None:
            prog.AddLinearConstraint(eq(p_WBs[0], pos_initial))
        if pos_target is not None:
            prog.AddLinearConstraint(eq(p_WBs[-1], pos_target))

        start = time.time()
        print("Starting to create SDP relaxation...")
        self.relaxed_prog, self.X, _ = create_sdp_relaxation(prog)
        end = time.time()
        print(
            f"Finished formulating relaxed problem. Elapsed time: {end - start} seconds"
        )

        # Retrieve original variables from X
        # We have n = 7*num_knot_points variables
        # This gives X dimensions (n+1,n+1)
        # where the 1 is added because the first element of x is 1.
        # This gives ((n+1)^2 - (n+1))/2 + (n+1) decision variables
        # (all entries - diagonal entries)/2 (because X symmetric) + add back diagonal)
        x = self.X[1:, 0]
        self.lam = x[0:num_knot_points]
        self.normal_forces = x[num_knot_points : 2 * num_knot_points]
        self.friction_forces = x[2 * num_knot_points : 3 * num_knot_points]
        self.cos_th = x[3 * num_knot_points : 4 * num_knot_points]
        self.sin_th = x[4 * num_knot_points : 5 * num_knot_points]
        self.p_WB_xs = x[5 * num_knot_points : 6 * num_knot_points]
        self.p_WB_ys = x[6 * num_knot_points : 7 * num_knot_points]

        # Keep variables
        # NOTE: Not in the original class definition
        self.p_WBs = np.hstack(
            [
                np.expand_dims(np.array([x, y]), 1)
                for x, y in zip(self.p_WB_xs, self.p_WB_ys)
            ]
        )
        self.f_c_Bs = np.hstack(
            [
                c_n * self.normal_vec + c_f * self.tangent_vec
                for c_n, c_f in zip(self.normal_forces, self.friction_forces)
            ]
        )
        self.p_c_Bs = np.hstack([l * self.pv1 + (1 - l) * self.pv2 for l in self.lam])

        self.mode_vars = ModeVars(
            self.cos_th,
            self.sin_th,
            self.p_WBs,
            self.p_c_Bs,
            self.f_c_Bs,
            self.time_in_mode,
        )

        self.num_variables = 7 * num_knot_points + 1  # TODO: 7 is hardcoded, fix this

    def get_spectrahedron(self) -> opt.Spectrahedron:
        # Variables should be the stacked columns of the lower
        # triangular part of the symmetric matrix from relaxed_prog, see
        # https://robotlocomotion.github.io/doxygen_cxx/classdrake_1_1solvers_1_1_mathematical_program.html#a8f718351922bc149cb6e7fa6d82288a5
        spectrahedron = opt.Spectrahedron(self.relaxed_prog)
        return spectrahedron

    def get_vars_from_gcs_vertex(
        self, gcs_vertex: opt.GraphOfConvexSets.Vertex
    ) -> ModeVars:
        x = gcs_vertex.x()[1 : self.num_variables + 2]

        lam = x[0 : self.num_knot_points]
        normal_forces = x[self.num_knot_points : 2 * self.num_knot_points]
        friction_forces = x[2 * self.num_knot_points : 3 * self.num_knot_points]
        cos_ths = x[3 * self.num_knot_points : 4 * self.num_knot_points]
        sin_ths = x[4 * self.num_knot_points : 5 * self.num_knot_points]
        p_WB_xs = x[5 * self.num_knot_points : 6 * self.num_knot_points]
        p_WB_ys = x[6 * self.num_knot_points : 7 * self.num_knot_points]

        p_WBs = np.hstack(
            [np.expand_dims(np.array([x, y]), 1) for x, y in zip(p_WB_xs, p_WB_ys)]
        )

        f_c_Bs = np.hstack(
            [
                c_n * self.normal_vec + c_f * self.tangent_vec
                for c_n, c_f in zip(normal_forces, friction_forces)
            ]
        )

        p_c_Bs = np.hstack([l * self.pv1 + (1 - l) * self.pv2 for l in lam])

        return ModeVars(
            cos_ths,
            sin_ths,
            p_WBs,
            p_c_Bs,
            f_c_Bs,
            self.time_in_mode,
        )

    def solve(self):
        print("Solving...")
        start = time.time()
        result = Solve(self.relaxed_prog)
        end = time.time()
        print(f"Solved in {end - start} seconds")
        assert result.is_success()
        print("Success!")

        lam_vals = result.GetSolution(self.lam)
        normal_forces_vals = result.GetSolution(self.normal_forces)
        friction_forces_vals = result.GetSolution(self.friction_forces)
        cos_th_vals = result.GetSolution(self.cos_th)
        sin_th_vals = result.GetSolution(self.sin_th)
        p_WB_xs_vals = result.GetSolution(self.p_WB_xs)
        p_WB_ys_vals = result.GetSolution(self.p_WB_ys)

        return (
            lam_vals,
            normal_forces_vals,
            friction_forces_vals,
            cos_th_vals,
            sin_th_vals,
            p_WB_xs_vals,
            p_WB_ys_vals,
        )


def plan_planar_pushing():
    experiment_number = 0
    if experiment_number == 0:
        th_initial = 0
        th_target = 0.8
        pos_initial = np.array([[0.0, 0.5]])
        pos_target = np.array([[-0.3, 0.2]])
    else:
        th_initial = 0
        th_target = 0.4
        pos_initial = np.array([[0.2, 0.2]])
        pos_target = np.array([[-0.18, 0.5]])

    num_knot_points = 8
    time_in_contact = 2

    MASS = 1.0
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
        contact_face_idx=0,
        end_time=time_in_contact,
        th_initial=th_initial,
        pos_initial=pos_initial,
        th_target=th_target,
        pos_target=pos_target,
    )

    result = Solve(contact_mode.relaxed_prog)
    assert result.is_success()
    print("Success!")

    vals = [contact_mode.mode_vars.eval_result(result)]

    DT = 0.01
    interpolate = False
    R_traj = sum(
        [val.get_R_traj(DT, interpolate=interpolate) for val in vals],
        [],
    )
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

    traj_length = len(R_traj)

    compute_violation = False
    if compute_violation:
        # NOTE: SHOULD BE MOVED!
        # compute quasi-static dynamic violation
        def _cross_2d(v1, v2):
            return (
                v1[0] * v2[1] - v1[1] * v2[0]
            )  # copied because the other one assumes the result is a np array, here it is just a scalar. clean up!

        quasi_static_violation = []
        for k in range(traj_length - 1):
            v_WB = (com_traj[k + 1] - com_traj[k]) / DT
            R_dot = (R_traj[k + 1] - R_traj[k]) / DT
            R = R_traj[k]
            omega_WB = R_dot.dot(R.T)[1, 0]
            f_c_B = force_traj[k]
            p_c_B = contact_pos_traj[k]
            com = com_traj[k]

            # Contact torques
            tau_c_B = _cross_2d(p_c_B - com, f_c_B)

            x_dot = np.concatenate([v_WB, [omega_WB]])
            wrench = np.concatenate(
                [f_c_B.flatten(), [tau_c_B]]
            )  # NOTE: Should fix not nice vector dimensions

            R_padded = np.zeros((3, 3))
            R_padded[2, 2] = 1
            R_padded[0:2, 0:2] = R
            violation = x_dot - R_padded.dot(A).dot(wrench)
            quasi_static_violation.append(violation)

        quasi_static_violation = np.vstack(quasi_static_violation)
        create_quasistatic_pushing_analysis(quasi_static_violation, num_knot_points)
        plt.show()

    CONTACT_COLOR = COLORS["dodgerblue4"]
    GRAVITY_COLOR = COLORS["blueviolet"]
    BOX_COLOR = COLORS["aquamarine4"]
    TABLE_COLOR = COLORS["bisque3"]
    FINGER_COLOR = COLORS["firebrick3"]
    TARGET_COLOR = COLORS["firebrick1"]

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
    VEL_VIZ_SCALE_CONSTANT = 0.005
    object_vel_viz = VisualizationForce2d(com_traj, CONTACT_COLOR, object_vel_traj / VEL_VIZ_SCALE_CONSTANT)  # type: ignore

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
        "--exp",
        help="Which experiment to run",
        type=int,
        default=0,
    )
    args = parser.parse_args()
    experiment_number = args.exp

    plan_planar_pushing()
