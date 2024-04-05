import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from pydrake.math import eq
from pydrake.solvers import MathematicalProgram, SnoptSolver, Solve

from planning_through_contact.experiments.utils import (
    get_default_plan_config,
    get_sugar_box,
)
from planning_through_contact.geometry.collision_geometry.box_2d import Box2d
from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    CollisionGeometry,
)
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.planar.planar_pushing_trajectory import (
    PlanarPushingTrajectory,
    SimplePlanarPushingTrajectory,
)
from planning_through_contact.geometry.utilities import (
    cross_2d,
    two_d_rotation_matrix_from_angle,
)
from planning_through_contact.planning.planar.planar_plan_config import (
    PlanarPushingStartAndGoal,
)
from planning_through_contact.tools.utils import (
    calc_displacements,
    evaluate_np_expressions_array,
)
from planning_through_contact.visualize.planar_pushing import (
    visualize_planar_pushing_trajectory,
)

num_time_steps = 10  # TODO: Change
dt = 0.1  # TODO: Change
end_time = num_time_steps * dt - dt
mu = 0.4

config = get_default_plan_config()
dynamics_config = config.dynamics_config

slider = get_sugar_box()

slider_initial_pose = PlanarPose(0, 0, 0)
slider_target_pose = PlanarPose(0.3, 0, np.pi / 2)
pusher_initial_pose = PlanarPose(-0.3, -0.3, 0)
pusher_target_pose = PlanarPose(-0.3, -0.3, 0)

config.start_and_goal = PlanarPushingStartAndGoal(
    slider_initial_pose, slider_target_pose, pusher_initial_pose, pusher_target_pose
)

NUM_DIMS = 2

prog = MathematicalProgram()
# Define decision variables
p_WBs = prog.NewContinuousVariables(num_time_steps, NUM_DIMS, "p_WBs")
p_BPs = prog.NewContinuousVariables(num_time_steps, NUM_DIMS, "p_BPs")
normal_forces = prog.NewContinuousVariables(num_time_steps - 1, 1, "lambda_n")
friction_forces = prog.NewContinuousVariables(num_time_steps - 1, 1, "lambda_f")
force_comps = np.hstack((normal_forces, friction_forces))

# r_WB_k = [cos_th; sin_th]
r_WBs = prog.NewContinuousVariables(num_time_steps, NUM_DIMS, "r_WBs")


# Create some convenience variables
def _two_d_rot_matrix_from_cos_sin(cos, sin) -> npt.NDArray:
    return np.array([[cos, -sin], [sin, cos]])


def _create_p_WP(p_WB, R_WB, p_BP):
    return p_WB + R_WB @ p_BP


R_WBs = [_two_d_rot_matrix_from_cos_sin(r_WB[0], r_WB[1]) for r_WB in r_WBs]
p_WPs = [
    _create_p_WP(p_WB, R_WB, p_BP) for p_WB, R_WB, p_BP in zip(p_WBs, R_WBs, p_BPs)
]


# Initial and target constraints on slider
p_WB_initial = slider_initial_pose.pos().flatten()
p_WB_target = slider_target_pose.pos().flatten()
R_WB_initial = two_d_rotation_matrix_from_angle(slider_initial_pose.theta)
R_WB_target = two_d_rotation_matrix_from_angle(slider_target_pose.theta)
p_WP_initial = _create_p_WP(p_WB_initial, R_WB_initial, p_BPs[0])
p_WP_target = _create_p_WP(p_WB_target, R_WB_target, p_BPs[-1])


def _add_slider_equal_pose_constraint(idx: int, target: PlanarPose):
    prog.AddLinearConstraint(eq(p_WBs[idx], target.pos().flatten()))
    prog.AddLinearConstraint(eq(r_WBs[idx], np.array([target.cos(), target.sin()])))


_add_slider_equal_pose_constraint(0, slider_target_pose)
_add_slider_equal_pose_constraint(-1, slider_initial_pose)

# Initial and target constraint on pusher
for c in eq(p_WP_initial, pusher_initial_pose.pos().flatten()):
    prog.AddLinearEqualityConstraint(c)
for c in eq(p_WP_target, pusher_target_pose.pos().flatten()):
    prog.AddLinearEqualityConstraint(c)

# SO(2) constraints
for cos_th, sin_th in r_WBs:
    prog.AddConstraint(cos_th**2 + sin_th**2 == 1)

# Dynamics
# Limit surface constants
c_f = dynamics_config.f_max**-2
c_tau = dynamics_config.tau_max**-2

_calc_contact_jacobian = lambda p_BP: slider.geometry.get_contact_jacobian(p_BP)  # type: ignore


def _calc_omega_WB(r_WB_curr, r_WB_next):
    R_WB_curr = _two_d_rot_matrix_from_cos_sin(r_WB_curr[0], r_WB_curr[1])
    R_WB_next = _two_d_rot_matrix_from_cos_sin(r_WB_next[0], r_WB_next[1])
    R_WB_dot = (R_WB_next - R_WB_curr) / dt
    # In 2D, omega_z = theta_dot will be at position (1,0) in R_dot * R'
    omega_WB = R_WB_dot.dot(R_WB_curr.T)[1, 0]
    return omega_WB


def _dynamics_constraint(vars: npt.NDArray) -> npt.NDArray:
    p_WB_curr = vars[0:2]
    p_WB_next = vars[2:4]
    r_WB_curr = vars[4:6]
    r_WB_next = vars[6:8]
    f_comps = vars[8:10]
    p_BP = vars[10:12]

    R_WB = _two_d_rot_matrix_from_cos_sin(r_WB_curr[0], r_WB_curr[1])
    v_WB = (p_WB_next - p_WB_curr) / dt
    omega_WB = _calc_omega_WB(r_WB_curr, r_WB_next)

    J_c = _calc_contact_jacobian(p_BP)
    gen_force = J_c.T @ f_comps

    f_c_B = gen_force[:2]
    trans_vel_constraint = v_WB - R_WB @ (c_f * f_c_B)

    tau_c_B = gen_force[2]
    ang_vel_constraint = omega_WB - tau_c_B

    return np.concatenate([trans_vel_constraint, [ang_vel_constraint]])


for k in range(num_time_steps - 1):
    p_WB_curr = p_WBs[k]
    p_WB_next = p_WBs[k + 1]
    r_WB_curr = r_WBs[k]
    r_WB_next = r_WBs[k + 1]
    f_comp_curr = force_comps[k]
    p_BP_curr = p_BPs[k]

    vars = np.concatenate(
        (p_WB_curr, p_WB_next, r_WB_curr, r_WB_next, f_comp_curr, p_BP_curr)
    )
    prog.AddConstraint(_dynamics_constraint, np.zeros((3,)), np.zeros((3,)), vars=vars)

# Enforce non-penetration
sdf = lambda pos: np.array([slider.geometry.get_signed_distance(pos)])  # type: ignore
for k in range(num_time_steps):
    prog.AddConstraint(sdf, np.zeros((1,)), np.ones((1,)) * np.inf, vars=p_BPs[k])


# Enforce friction cone
for lambda_n, lambda_f in force_comps:
    prog.AddLinearConstraint(lambda_n >= 0)
    prog.AddLinearConstraint(lambda_f <= mu * lambda_n)
    prog.AddLinearConstraint(lambda_f >= -mu * lambda_n)


# # Complementarity constraints
# for k in range(num_time_steps - 1):
#     breakpoint()


# Create initial guess as straight line interpolation
def _interpolate_traj_1d(
    initial_val: float, target_val: float
) -> npt.NDArray[np.float64]:
    xs = np.array([0, end_time])
    ys = np.array([initial_val, target_val])
    x_interpolate = np.arange(0, end_time + dt, dt)
    y_interpolated = np.interp(x_interpolate, xs, ys)
    return y_interpolated


def _interpolate_traj(
    initial_val: npt.NDArray[np.float64], target_val: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    if len(initial_val.shape) == 2:
        initial_val = initial_val.flatten()
        target_val = target_val.flatten()

    num_dims = initial_val.shape[0]
    trajs = np.vstack(
        [
            _interpolate_traj_1d(initial_val[dim], target_val[dim])
            for dim in range(num_dims)
        ]
    ).T  # (num_time_steps, num_dims)
    return trajs


th_interpolated = _interpolate_traj_1d(
    slider_initial_pose.theta, slider_target_pose.theta
)
r_WBs_initial_guess = [np.array([np.cos(th), np.sin(th)]) for th in th_interpolated]
prog.SetInitialGuess(r_WBs, r_WBs_initial_guess)  # type: ignore

p_WBs_interpolated = _interpolate_traj(
    slider_initial_pose.pos(), slider_target_pose.pos()
)
prog.SetInitialGuess(p_WBs, p_WBs_interpolated)  # type: ignore

p_BPs_interpolated = _interpolate_traj(
    pusher_initial_pose.pos(), pusher_target_pose.pos()
)
prog.SetInitialGuess(p_BPs, p_BPs_interpolated)  # type: ignore

# Solve program
snopt = SnoptSolver()
result = snopt.Solve(prog)  # type: ignore
assert result.is_success()

traj = SimplePlanarPushingTrajectory(
    result.GetSolution(p_WBs),
    [evaluate_np_expressions_array(R_WB, result) for R_WB in R_WBs],
    evaluate_np_expressions_array(p_WPs, result),  # type: ignore
    evaluate_np_expressions_array(f_c_Ws, result),  # type: ignore
    dt,
    config,
)

visualize_planar_pushing_trajectory(
    traj,  # type: ignore
    save=True,
    # show=True,
    filename=f"direct_trajopt_test",
    visualize_knot_points=True,
)
