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
    NUM_KNOT_POINTS = 4
    END_TIME = 1
    CONTACT_FACE_IDX = 0
    FRICTION_COEFF = 0.5

    TH_INITIAL = 0.0
    TH_TARGET = 0.2

    A = np.diag([1.0, 1.0, 1.0])  # TODO: change

    dt = END_TIME / NUM_KNOT_POINTS

    prog = MathematicalProgram()

    box = EquilateralPolytope2d(
        actuated=False,
        name="Slider",
        mass=1.0,
        vertex_distance=0.2,
        num_vertices=4,
    )

    contact_face = PolytopeContactLocation(
        pos=ContactLocation.FACE, idx=CONTACT_FACE_IDX
    )

    # Contact positions
    lams = prog.NewContinuousVariables(NUM_KNOT_POINTS, "lam")
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

    # TODO: remove
    #    # Box rotation relative to world frame
    #    cos_ths = prog.NewContinuousVariables(NUM_KNOT_POINTS, "cos_th")
    #    sin_ths = prog.NewContinuousVariables(NUM_KNOT_POINTS, "sin_th")
    #    r_WBs = [np.array([c, s]) for c, s in zip(cos_ths, sin_ths)]

    # r_WB_dots = forward_differences(r_WBs, dt)

    #    sq_angular_vels = sum(
    #        [r_WB.T.dot(r_WB) for r_WB in r_WBs]
    #    )  # See paper for derivation of why this is equal to squared ang vels

    theta_WBs = prog.NewContinuousVariables(NUM_KNOT_POINTS, "theta")

    # Box position relative to world frame
    p_WB_xs = prog.NewContinuousVariables(NUM_KNOT_POINTS, "p_WB_x")
    p_WB_ys = prog.NewContinuousVariables(NUM_KNOT_POINTS, "p_WB_y")
    p_WBs = [np.array([x, y]) for x, y in zip(p_WB_xs, p_WB_ys)]

    # Compute velocities
    v_WBs = forward_differences(p_WBs, dt)
    omega_WBs = forward_differences(theta_WBs, dt)

    # # Friction cone constraints
    for c_n in normal_forces:
        prog.AddLinearConstraint(c_n >= 0)
    # for c_n, c_f in zip(normal_forces, friction_forces):
    #     prog.AddLinearConstraint(c_f <= FRICTION_COEFF * c_n)
    #     prog.AddLinearConstraint(c_f >= -FRICTION_COEFF * c_n)

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

    # Minimize kinetic energy through squared velocities
    sq_linear_vels = sum([v_WB.T.dot(v_WB) for v_WB in v_WBs])
    sq_angular_vels = sum([omega**2 for omega in omega_WBs])
    prog.AddQuadraticCost(sq_linear_vels)
    prog.AddQuadraticCost(sq_angular_vels)

    # Initial conditions
    prog.AddConstraint(theta_WBs[0] == TH_INITIAL)
    prog.AddConstraint(theta_WBs[-1] == TH_TARGET)

    result = Solve(prog)
    assert result.is_success()

    breakpoint()


if __name__ == "__main__":
    plan_planar_pushing()
