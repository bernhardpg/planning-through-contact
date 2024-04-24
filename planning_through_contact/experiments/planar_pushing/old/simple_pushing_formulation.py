import argparse
from typing import List, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from pydrake.math import eq
from pydrake.solvers import MathematicalProgram, Solve
from pydrake.trajectories import PiecewisePolynomial, PiecewiseQuaternionSlerp

from convex_relaxation.sdp import create_sdp_relaxation
from geometry.two_d.contact.types import ContactLocation
from geometry.two_d.equilateral_polytope_2d import EquilateralPolytope2d
from geometry.two_d.rigid_body_2d import PolytopeContactLocation
from geometry.utilities import cross_2d
from tools.types import NpExpressionArray
from visualize.analysis import create_quasistatic_pushing_analysis
from visualize.colors import COLORS
from visualize.visualizer_2d import (
    VisualizationForce2d,
    VisualizationPoint2d,
    VisualizationPolygon2d,
    Visualizer2d,
)

T = TypeVar("T")


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


def plan_planar_pushing(
    contact_face_idx,
    th_initial,
    th_target,
    pos_initial,
    pos_target,
    num_knot_points,
    num_vertices,
):
    FRICTION_COEFF = 0.5
    G = 9.81
    MASS = 1.0
    DIST_TO_CORNERS = 0.2

    f_max = FRICTION_COEFF * G * MASS
    tau_max = f_max * DIST_TO_CORNERS

    A = np.diag(
        [1 / f_max**2, 1 / f_max**2, 1 / tau_max**2]
    )  # Ellipsoidal Limit surface approximation

    dt = end_time / num_knot_points

    prog = MathematicalProgram()

    polytope = EquilateralPolytope2d(
        actuated=False,
        name="Slider",
        mass=MASS,
        vertex_distance=DIST_TO_CORNERS,
        num_vertices=num_vertices,
    )

    contact_face = PolytopeContactLocation(
        pos=ContactLocation.FACE, idx=contact_face_idx
    )

    # Contact positions
    lams = prog.NewContinuousVariables(num_knot_points, "lam")
    for lam in lams:
        prog.AddLinearConstraint(lam >= 0)
        prog.AddLinearConstraint(lam <= 1)

    pv1, pv2 = polytope.get_proximate_vertices_from_location(contact_face)
    p_c_Bs = [lam * pv1 + (1 - lam) * pv2 for lam in lams]

    # Contact forces
    normal_forces = prog.NewContinuousVariables(num_knot_points, "c_n")
    friction_forces = prog.NewContinuousVariables(num_knot_points, "c_f")
    normal_vec, tangent_vec = polytope.get_norm_and_tang_vecs_from_location(
        contact_face
    )
    f_c_Bs = [
        c_n * normal_vec + c_f * tangent_vec
        for c_n, c_f in zip(normal_forces, friction_forces)
    ]

    # Rotations
    theta_WBs = prog.NewContinuousVariables(num_knot_points, "theta")

    # Box position relative to world frame
    p_WB_xs = prog.NewContinuousVariables(num_knot_points, "p_WB_x")
    p_WB_ys = prog.NewContinuousVariables(num_knot_points, "p_WB_y")
    p_WBs = [np.array([x, y]) for x, y in zip(p_WB_xs, p_WB_ys)]  # TODO: rename!

    # Compute velocities
    v_WBs = forward_differences(p_WBs, dt)
    omega_WBs = forward_differences(theta_WBs, dt)
    v_c_Bs = forward_differences(p_c_Bs, dt)

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
        f_c_B = (f_c_Bs[k] + f_c_Bs[k + 1]) / 2
        p_c_B = (p_c_Bs[k] + p_c_Bs[k + 1]) / 2

        # Contact torques
        tau_c_B = cross_2d(p_c_B, f_c_B)

        x_dot = np.concatenate([v_WB, [omega_WB]])
        wrench = np.concatenate(
            [f_c_B.flatten(), [tau_c_B]]
        )  # NOTE: Should fix not nice vector dimensions

        quasi_static_dynamic_constraint = eq(x_dot, A.dot(wrench))
        for row in quasi_static_dynamic_constraint:
            prog.AddConstraint(row)

    # Ensure sticking on the contact point
    for v_c_B in v_c_Bs:
        prog.AddLinearConstraint(eq(v_c_B, 0))

    # Minimize kinetic energy through squared velocities
    sq_linear_vels = sum([v_WB.T.dot(v_WB) for v_WB in v_WBs])
    sq_angular_vels = sum([omega**2 for omega in omega_WBs])
    prog.AddQuadraticCost(sq_linear_vels)
    prog.AddQuadraticCost(sq_angular_vels)

    # Initial conditions
    prog.AddConstraint(theta_WBs[0] == th_initial)
    prog.AddConstraint(theta_WBs[-1] == th_target)

    def create_R(th: float) -> npt.NDArray[np.float64]:
        return np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])

    R_WB_I = create_R(th_initial)
    R_WB_T = create_R(th_target)
    p_WB_I = R_WB_I.dot(p_WBs[0])
    p_WB_T = R_WB_T.dot(p_WBs[-1])
    prog.AddLinearConstraint(eq(p_WB_I, pos_initial))
    prog.AddLinearConstraint(eq(p_WB_T, pos_target))

    import time

    start = time.time()
    print("Starting to create SDP relaxation...")
    relaxed_prog, X, basis = create_sdp_relaxation(prog)
    end = time.time()
    print(f"Finished formulating relaxed problem. Elapsed time: {end - start} seconds")

    print("Solving...")
    start = time.time()
    result = Solve(relaxed_prog)
    end = time.time()
    print(f"Solved in {end - start} seconds")
    assert result.is_success()
    print("Success!")

    # Extract valus
    x_val = result.GetSolution(X[1:, 0])
    lam_vals = x_val[0:num_knot_points]
    normal_forces_vals = x_val[num_knot_points : 2 * num_knot_points]
    friction_forces_vals = x_val[2 * num_knot_points : 3 * num_knot_points]
    theta_vals = x_val[3 * num_knot_points : 4 * num_knot_points]
    p_WB_xs_vals = x_val[4 * num_knot_points : 5 * num_knot_points]
    p_WB_ys_vals = x_val[5 * num_knot_points : 6 * num_knot_points]

    # Reconstruct quantities
    p_c_Bs_vals = np.hstack([lam * pv1 + (1 - lam) * pv2 for lam in lam_vals]).T
    f_c_Bs_vals = np.hstack(
        [
            c_n * normal_vec + c_f * tangent_vec
            for c_n, c_f in zip(normal_forces_vals, friction_forces_vals)
        ]
    ).T
    p_WBs_vals = np.vstack(
        [np.array([x, y]) for x, y in zip(p_WB_xs_vals, p_WB_ys_vals)]
    )  # TODO: rename!

    R_WBs_vals = [
        np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        for th in theta_vals
    ]

    box_com_traj = np.vstack(
        [R_WB.dot(p_body) for R_WB, p_body in zip(R_WBs_vals, p_WBs_vals)]
    )

    contact_pos_in_W = np.vstack(
        [
            p_WB + R_WB.dot(p_c_B)
            for p_WB, R_WB, p_c_B in zip(box_com_traj, R_WBs_vals, p_c_Bs_vals)
        ]
    )

    contact_force_in_W = np.vstack(
        [R_WB.dot(f_c_B) for R_WB, f_c_B in zip(R_WBs_vals, f_c_Bs_vals)]
    )

    DT = 0.01

    # Interpolate quantities
    R_traj = interpolate_so2_using_slerp(R_WBs_vals, 0, end_time, DT)
    com_traj = interpolate_w_first_order_hold(box_com_traj, 0, end_time, DT)
    force_traj = interpolate_w_first_order_hold(contact_force_in_W, 0, end_time, DT)
    contact_pos_traj = interpolate_w_first_order_hold(contact_pos_in_W, 0, end_time, DT)
    theta_traj = interpolate_w_first_order_hold(
        theta_vals.reshape((-1, 1)), 0, end_time, DT
    )

    traj_length = len(R_traj)

    # NOTE: SHOULD BE MOVED!
    # compute quasi-static dynamic violation
    def _cross_2d(v1, v2):
        return (
            v1[0] * v2[1] - v1[1] * v2[0]
        )  # copied because the other one assumes the result is a np array, here it is just a scalar. clean up!

    quasi_static_violation = []
    for k in range(traj_length - 1):
        v_WB = (com_traj[k + 1] - com_traj[k]) / DT
        omega_WB = (theta_traj[k + 1] - theta_traj[k]) / DT
        f_c_B = force_traj[k]
        p_c_B = contact_pos_traj[k]
        com = com_traj[k]
        # omega_WB = ((R_traj[k + 1] - R_traj[k]) / DT) * R_traj[k].T

        # Contact torques
        tau_c_B = _cross_2d(p_c_B - com, f_c_B)

        x_dot = np.concatenate([v_WB, omega_WB])
        wrench = np.concatenate(
            [f_c_B.flatten(), [tau_c_B]]
        )  # NOTE: Should fix not nice vector dimensions

        violation = x_dot - A.dot(wrench)
        quasi_static_violation.append(violation)

    quasi_static_violation = np.vstack(quasi_static_violation)
    # create_quasistatic_pushing_analysis(quasi_static_violation, num_knot_points)
    # plt.show()

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
        polytope,
        BOX_COLOR,
    )

    # NOTE: I don't really need the entire trajectory here, but leave for now
    target_viz = VisualizationPolygon2d.from_trajs(
        com_traj,
        flattened_rotation,
        polytope,
        TARGET_COLOR,
    )

    com_points_viz = VisualizationPoint2d(com_traj, GRAVITY_COLOR)  # type: ignore
    contact_point_viz = VisualizationPoint2d(contact_pos_traj, FINGER_COLOR)  # type: ignore
    contact_force_viz = VisualizationForce2d(contact_pos_traj, CONTACT_COLOR, force_traj)  # type: ignore

    viz = Visualizer2d()
    FRAMES_PER_SEC = len(R_traj) / end_time
    viz.visualize(
        [com_points_viz, contact_point_viz],
        [contact_force_viz],
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

    if experiment_number == 0:
        contact_face_idx = 0
        num_knot_points = 10
        end_time = 3
        num_vertices = 4

        th_initial = np.pi / 4
        th_target = np.pi / 4 - 0.5
        pos_initial = np.array([[0.0, 0.5]])
        pos_target = np.array([[-0.1, 0.2]])
    elif experiment_number == 1:
        num_vertices = 3
        contact_face_idx = 0
        num_knot_points = 10
        end_time = 3

        th_initial = np.pi / 4
        th_target = np.pi / 4 + 0.5
        pos_initial = np.array([[0.0, 0.5]])
        pos_target = np.array([[-0.1, 0.2]])

    elif experiment_number == 2:
        num_vertices = 5
        contact_face_idx = 0
        num_knot_points = 10
        end_time = 3

        th_initial = np.pi / 4 + np.pi / 2 - 0.7
        th_target = np.pi / 4 + np.pi / 2
        pos_initial = np.array([[-0.2, 0.3]])
        pos_target = np.array([[0.3, 0.5]])

    plan_planar_pushing(
        contact_face_idx,
        th_initial,
        th_target,
        pos_initial,
        pos_target,
        num_knot_points,
        num_vertices,
    )
