import numpy as np
import numpy.typing as npt
from pydrake.math import eq
from pydrake.solvers import MathematicalProgram, SnoptSolver, Solve

from planning_through_contact.experiments.utils import get_sugar_box
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.utilities import two_d_rotation_matrix_from_angle

num_time_steps = 10
dt = 0.1
end_time = num_time_steps * dt - dt

NUM_DIMS = 2

prog = MathematicalProgram()

# Define decision variables
p_WBs = prog.NewContinuousVariables(num_time_steps, NUM_DIMS, "p_WBs")
p_BPs = prog.NewContinuousVariables(num_time_steps, NUM_DIMS, "p_BPs")
f_c_Bs = prog.NewContinuousVariables(num_time_steps, NUM_DIMS, "f_c_Bs")
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

slider = get_sugar_box()

slider_initial_pose = PlanarPose(0, 0, 0)
slider_target_pose = PlanarPose(0.3, 0, np.pi / 2)
pusher_initial_pose = PlanarPose(-0.3, -0.3, 0)
pusher_target_pose = PlanarPose(-0.3, -0.3, 0)


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

snopt = SnoptSolver()


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

result = snopt.Solve(prog)  # type: ignore
assert result.is_success()

breakpoint()
