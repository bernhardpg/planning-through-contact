from typing import List, TypeVar

import numpy as np
import numpy.typing as npt
from pydrake.math import eq
from pydrake.solvers import MathematicalProgram, Solve

from convex_relaxation.sdp import create_sdp_relaxation
from geometry.two_d.contact.types import ContactLocation
from geometry.two_d.equilateral_polytope_2d import EquilateralPolytope2d
from geometry.two_d.rigid_body_2d import PolytopeContactLocation
from geometry.utilities import cross_2d
from tools.types import NpExpressionArray
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


def plan_planar_pushing():
    NUM_KNOT_POINTS = 10
    END_TIME = 3
    CONTACT_FACE_IDX = 0
    FRICTION_COEFF = 0.5
    NUM_DIMS = 2
    G = 9.81
    MASS = 1.0
    DIST_TO_CORNERS = 0.2

    TH_INITIAL = np.pi / 4
    TH_TARGET = np.pi / 4 - 0.5

    POS_INITIAL = np.array([[0.0, 0.5]])
    POS_TARGET = np.array([[-0.1, 0.2]])

    f_max = FRICTION_COEFF * G * MASS
    tau_max = f_max * DIST_TO_CORNERS

    A = np.diag(
        [1 / f_max**2, 1 / f_max**2, 1 / tau_max**2]
    )  # Ellipsoidal Limit surface approximation

    dt = END_TIME / NUM_KNOT_POINTS

    prog = MathematicalProgram()

    box = EquilateralPolytope2d(
        actuated=False,
        name="Slider",
        mass=MASS,
        vertex_distance=DIST_TO_CORNERS,
        num_vertices=4,
    )

    contact_face = PolytopeContactLocation(
        pos=ContactLocation.FACE, idx=CONTACT_FACE_IDX
    )

    # Contact positions
    lams = prog.NewContinuousVariables(NUM_KNOT_POINTS, "lam")
    for lam in lams:
        prog.AddLinearConstraint(lam >= 0)
        prog.AddLinearConstraint(lam <= 1)

    pv1, pv2 = box.get_proximate_vertices_from_location(contact_face)
    p_c_Bs = [lam * pv1 + (1 - lam) * pv2 for lam in lams]

    # Contact forces
    normal_forces = prog.NewContinuousVariables(NUM_KNOT_POINTS, "c_n")
    friction_forces = prog.NewContinuousVariables(NUM_KNOT_POINTS, "c_f")
    normal_vec, tangent_vec = box.get_norm_and_tang_vecs_from_location(contact_face)
    f_c_Bs = [
        c_n * normal_vec + c_f * tangent_vec
        for c_n, c_f in zip(normal_forces, friction_forces)
    ]

    # Contact torques
    tau_c_Bs = [cross_2d(p_c_B, f_c_B) for p_c_B, f_c_B in zip(p_c_Bs, f_c_Bs)]

    # Rotations
    theta_WBs = prog.NewContinuousVariables(NUM_KNOT_POINTS, "theta")

    # Box position relative to world frame
    p_WB_xs = prog.NewContinuousVariables(NUM_KNOT_POINTS, "p_WB_x")
    p_WB_ys = prog.NewContinuousVariables(NUM_KNOT_POINTS, "p_WB_y")
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
    for v_WB, omega_WB, f_c_B, tau_c_B in zip(
        v_WBs, omega_WBs, f_c_Bs, tau_c_Bs
    ):  # NOTE: This will not add any dynamic constraints to the final forces and torques!
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
    prog.AddConstraint(theta_WBs[0] == TH_INITIAL)
    prog.AddConstraint(theta_WBs[-1] == TH_TARGET)

    def create_R(th: float) -> npt.NDArray[np.float64]:
        return np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])

    R_WB_I = create_R(TH_INITIAL)
    R_WB_T = create_R(TH_TARGET)
    p_WB_I = R_WB_I.dot(p_WBs[0])
    p_WB_T = R_WB_T.dot(p_WBs[-1])
    prog.AddLinearConstraint(eq(p_WB_I, POS_INITIAL))
    prog.AddLinearConstraint(eq(p_WB_T, POS_TARGET))

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
    lam_vals = x_val[0:NUM_KNOT_POINTS]
    normal_forces_vals = x_val[NUM_KNOT_POINTS : 2 * NUM_KNOT_POINTS]
    friction_forces_vals = x_val[2 * NUM_KNOT_POINTS : 3 * NUM_KNOT_POINTS]
    theta_vals = x_val[3 * NUM_KNOT_POINTS : 4 * NUM_KNOT_POINTS]
    p_WB_xs_vals = x_val[4 * NUM_KNOT_POINTS : 5 * NUM_KNOT_POINTS]
    p_WB_ys_vals = x_val[5 * NUM_KNOT_POINTS : 6 * NUM_KNOT_POINTS]

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

    rotation = np.vstack(
        [
            np.array(
                [[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]
            ).flatten()  # NOTE: This is the expected format by the visualizer
            for th in theta_vals
        ]
    )

    com = np.vstack(
        [
            R_WB.reshape((NUM_DIMS, NUM_DIMS)).dot(p_body)
            for R_WB, p_body in zip(rotation, p_WBs_vals)
        ]
    )

    contact_pos_in_W = np.vstack(
        [
            p_WB + R_WB.reshape((NUM_DIMS, NUM_DIMS)).dot(p_c_B)
            for p_WB, R_WB, p_c_B in zip(com, rotation, p_c_Bs_vals)
        ]
    )

    contact_force_in_W = np.vstack(
        [
            R_WB.reshape((NUM_DIMS, NUM_DIMS)).dot(f_c_B)
            for R_WB, f_c_B in zip(rotation, f_c_Bs_vals)
        ]
    )

    CONTACT_COLOR = COLORS["dodgerblue4"]
    GRAVITY_COLOR = COLORS["blueviolet"]
    BOX_COLOR = COLORS["aquamarine4"]
    TABLE_COLOR = COLORS["bisque3"]
    FINGER_COLOR = COLORS["firebrick3"]

    box_viz = VisualizationPolygon2d.from_trajs(
        com,
        rotation,
        box,
        BOX_COLOR,
    )

    com_points_viz = VisualizationPoint2d(com, GRAVITY_COLOR)  # type: ignore
    contact_point_viz = VisualizationPoint2d(contact_pos_in_W, FINGER_COLOR)  # type: ignore
    contact_force_viz = VisualizationForce2d(contact_pos_in_W, CONTACT_COLOR, contact_force_in_W)  # type: ignore

    viz = Visualizer2d()
    FRAMES_PER_SEC = NUM_KNOT_POINTS / END_TIME
    viz.visualize(
        [com_points_viz, contact_point_viz],
        [contact_force_viz],
        [box_viz],
        FRAMES_PER_SEC,
    )


if __name__ == "__main__":
    plan_planar_pushing()
