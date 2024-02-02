from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import List, Optional, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pydrake.geometry.optimization as opt
from pydrake.solvers import MathematicalProgramResult
from pydrake.systems.primitives import VectorLog

from planning_through_contact.experiments.ablation_study.planar_pushing_ablation import (
    SingleRunResult,
)
from planning_through_contact.geometry.bezier import BezierCurve
from planning_through_contact.geometry.in_plane.contact_pair import (
    ContactFrameConstraints,
)
from planning_through_contact.geometry.planar.face_contact import (
    FaceContactMode,
    FaceContactVariables,
)
from planning_through_contact.geometry.planar.planar_pushing_path import (
    PlanarPushingPath,
)
from planning_through_contact.geometry.planar.planar_pushing_trajectory import (
    PlanarPushingTrajectory,
)
from planning_through_contact.geometry.utilities import cross_2d
from planning_through_contact.planning.planar.planar_plan_config import (
    PlanarSolverParams,
)


def save_gcs_graph_diagram(
    gcs: opt.GraphOfConvexSets,
    filepath: Path,
    result: Optional[MathematicalProgramResult] = None,
) -> None:
    graphviz = gcs.GetGraphvizString(result, precision=1)
    import pydot

    data = pydot.graph_from_dot_data(graphviz)[0]  # type: ignore
    data.write_svg(str(filepath))


@dataclass
class PlanarPushingLog:
    t: npt.NDArray[np.float64]
    x: npt.NDArray[np.float64]
    y: npt.NDArray[np.float64]
    theta: npt.NDArray[np.float64]
    lam: npt.NDArray[np.float64]
    c_n: npt.NDArray[np.float64]
    c_f: npt.NDArray[np.float64]
    lam_dot: npt.NDArray[np.float64]

    @classmethod
    def from_np(
        cls,
        t: npt.NDArray[np.float64],
        state_np_array: npt.NDArray[np.float64],
        control_np_array: npt.NDArray[np.float64],
    ) -> "PlanarPushingLog":
        x = state_np_array[0, :]
        y = state_np_array[1, :]
        theta = state_np_array[2, :]
        if state_np_array.shape[0] == 3:
            # Padding state since we didn't log lam
            lam = np.zeros_like(x)
        else:
            lam = state_np_array[3, :]

        c_n = control_np_array[0, :]
        c_f = control_np_array[1, :]
        lam_dot = control_np_array[2, :]
        return cls(t, x, y, theta, lam, c_n, c_f, lam_dot)

    @classmethod
    def from_log(
        cls,
        state_log: VectorLog,
        control_log: VectorLog,
    ) -> "PlanarPushingLog":
        t = state_log.sample_times()
        state_np_array = state_log.data()
        control_np_array = control_log.data()
        return cls.from_np(t, state_np_array, control_np_array)

    @classmethod
    def from_pose_vector_log(
        cls,
        pose_vector_log: VectorLog,
    ) -> "PlanarPushingLog":
        t = pose_vector_log.sample_times()
        state_np_array = pose_vector_log.data()
        PAD_VAL = 0
        single_row_pad = np.ones_like(state_np_array[0, :]) * PAD_VAL
        if state_np_array.shape[0] == 3:
            # Padding state since we didn't log lam
            state_np_array = np.vstack((state_np_array, single_row_pad))
        elif state_np_array.shape[0] == 2:
            # Padding state since we didn't log theta and lam
            state_np_array = np.vstack(
                (
                    state_np_array,
                    single_row_pad,
                    single_row_pad,
                )
            )
        control_np_array = np.ones((3, len(t))) * PAD_VAL
        return cls.from_np(t, state_np_array, control_np_array)


@dataclass
class CombinedPlanarPushingLogs:
    pusher_actual: PlanarPushingLog
    slider_actual: PlanarPushingLog
    pusher_desired: PlanarPushingLog
    slider_desired: PlanarPushingLog


def plot_planar_pushing_logs(
    state_log: VectorLog,
    state_log_desired: VectorLog,
    control_log: VectorLog,
    control_log_desired: VectorLog,
) -> None:
    actual = PlanarPushingLog.from_log(state_log, control_log)
    desired = PlanarPushingLog.from_log(state_log_desired, control_log_desired)

    plot_planar_pushing_trajectory(actual, desired)


def plot_control_sols_vs_time(control_log: List[np.ndarray], suffix: str = "", save_dir: Optional[str] = None) -> None:
    # Convert the list to a numpy array for easier manipulation
    control_log_array = np.array(control_log)

    # Prepare data for plotting
    timesteps = np.arange(control_log_array.shape[0])
    prediction_horizons = np.arange(control_log_array.shape[1])

    # Create a meshgrid for timesteps and prediction_horizons
    T, P = np.meshgrid(prediction_horizons, timesteps)  # Note the change in the order

    # Initialize a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot each control input
    for i, label in enumerate(["c_n", "c_f", "lam_dot"]):
        # Extract the control data for plotting
        Z = control_log_array[:, :, i]

        # Ensure Z has the same shape as T and P
        # Z might need to be transposed depending on how control_log_array is structured
        Z = Z.T if Z.shape != T.shape else Z

        ax.plot_surface(T, P, Z, label=label, alpha=0.7)

    # Adding labels
    ax.set_xlabel("Prediction Horizon")
    ax.set_ylabel("Timestep")
    ax.set_zlabel("Control Inputs")

    # Adding title
    ax.set_title("3D Control Inputs Plot")

    # Workaround for legend in 3D plot
    from matplotlib.lines import Line2D

    custom_lines = [
        Line2D([0], [0], linestyle="none", marker="_", color="blue", markersize=10),
        Line2D([0], [0], linestyle="none", marker="_", color="orange", markersize=10),
        Line2D([0], [0], linestyle="none", marker="_", color="green", markersize=10),
    ]
    ax.legend(custom_lines, ["c_n", "c_f", "lam_dot"])

    # Show plot
    plt.tight_layout()
    file_name = f"planar_pushing_control_sols{suffix}.pdf"
    file_path = f"{save_dir}/{file_name}" if save_dir else file_name
    plt.savefig(file_path)


def plot_cost(cost_log: List[float], suffix: str = "", save_dir: Optional[str] = None) -> None:
    plt.figure()
    plt.plot(cost_log)
    plt.title("Cost vs. timestep")
    plt.xlabel("timestep")
    plt.ylabel("Cost")
    plt.tight_layout()

    file_name = f"planar_pushing_cost{suffix}.pdf"
    file_path = f"{save_dir}/{file_name}" if save_dir else file_name
    plt.savefig(file_path)


def plot_velocities(
    desired_vel_log: List[npt.NDArray],
    commanded_vel_log: List[npt.NDArray],
    suffix: str = "",
) -> None:
    plt.figure()
    # velocity has x and y dimensions
    desired_vel_log_array = np.array(desired_vel_log)
    commanded_vel_log_array = np.array(commanded_vel_log)
    timesteps = np.arange(desired_vel_log_array.shape[0])
    plt.plot(timesteps, desired_vel_log_array[:, 0], label="desired x vel")
    plt.plot(timesteps, desired_vel_log_array[:, 1], label="desired y vel")
    plt.plot(timesteps, commanded_vel_log_array[:, 0], label="commanded x vel")
    plt.plot(timesteps, commanded_vel_log_array[:, 1], label="commanded y vel")
    plt.legend()
    plt.title("Desired and commanded velocities vs. timestep")
    plt.xlabel("timestep")
    plt.ylabel("Velocity")
    plt.tight_layout()
    plt.savefig(f"planar_pushing_velocities{suffix}.png")


def plot_and_save_planar_pushing_logs_from_sim(
    pusher_pose_vector_log: VectorLog,
    slider_pose_vector_log: VectorLog,
    control_log: VectorLog,
    control_desired_log: VectorLog,
    pusher_pose_vector_log_desired: VectorLog,
    slider_pose_vector_log_desired: VectorLog,
    save_dir: Optional[str] = None,
) -> None:
    pusher_actual = PlanarPushingLog.from_pose_vector_log(pusher_pose_vector_log)
    slider_actual = PlanarPushingLog.from_log(slider_pose_vector_log, control_log)
    pusher_desired = PlanarPushingLog.from_pose_vector_log(
        pusher_pose_vector_log_desired
    )
    slider_desired = PlanarPushingLog.from_log(
        slider_pose_vector_log_desired,
        control_desired_log,
    )
    # Save the logs
    combined = CombinedPlanarPushingLogs(
        pusher_actual=pusher_actual,
        slider_actual=slider_actual,
        pusher_desired=pusher_desired,
        slider_desired=slider_desired,
    )

    with open(f"{save_dir}/combined_planar_pushing_logs.pkl", "wb") as f:
        pickle.dump(combined, f)

    plot_planar_pushing_trajectory(
        slider_actual,
        slider_desired,
        suffix="_slider",
        plot_lam=False,
        save_dir=save_dir,
    )
    plot_planar_pushing_trajectory(
        pusher_actual,
        pusher_desired,
        suffix="_pusher",
        plot_lam=False,
        plot_control=False,
        save_dir=save_dir,
    )


def plot_planar_pushing_trajectory(
    actual: PlanarPushingLog,
    desired: PlanarPushingLog,
    plot_control: bool = True,
    plot_lam: bool = True,
    suffix: str = "",
    save_dir: Optional[str] = None,
) -> None:
    # State plot
    fig, axes = plt.subplots(nrows=4 if plot_lam else 3, ncols=1, figsize=(8, 8))
    MIN_AXIS_SIZE = 0.1

    pos = np.vstack((actual.x, actual.y))
    max_pos_change = max(np.ptp(np.linalg.norm(pos, axis=0)), MIN_AXIS_SIZE) * 1.3
    # Note: this calculation doesn't center the plot on the right value, so
    # the line might not be visible

    axes[0].plot(actual.t, actual.x, label="Actual")
    axes[0].plot(actual.t, desired.x, linestyle="--", label="Desired")
    axes[0].set_title("x")
    axes[0].legend()
    # axes[0].set_ylim(-max_pos_change, max_pos_change)

    axes[1].plot(actual.t, actual.y, label="Actual")
    axes[1].plot(actual.t, desired.y, linestyle="--", label="Desired")
    axes[1].set_title("y")
    axes[1].legend()
    axes[1].set_ylim(-max_pos_change, max_pos_change)

    th_change = max(np.ptp(actual.theta), MIN_AXIS_SIZE) * 2.0  # type: ignore

    axes[2].plot(actual.t, actual.theta, label="Actual")
    axes[2].plot(actual.t, desired.theta, linestyle="--", label="Desired")
    axes[2].set_title("theta")
    axes[2].legend()
    # axes[2].set_ylim(-th_change, th_change)

    if plot_lam:
        axes[3].plot(actual.t, actual.lam, label="Actual")
        axes[3].plot(actual.t, desired.lam, linestyle="--", label="Desired")
        axes[3].set_title("lam")
        axes[3].legend()
        axes[3].set_ylim(0, 1)

    plt.tight_layout()
    file_name = f"planar_pushing_states{suffix}.pdf"
    file_path = f"{save_dir}/{file_name}" if save_dir else file_name
    plt.savefig(file_path)

    # State Error plot
    fig, axes = plt.subplots(nrows=4 if plot_lam else 3, ncols=1, figsize=(8, 8))
    MIN_AXIS_SIZE = 0.1

    x_error = actual.x - desired.x
    y_error = actual.y - desired.y
    pos = np.vstack((x_error, y_error))
    # max_pos_change = max(np.ptp(np.linalg.norm(pos, axis=0)), MIN_AXIS_SIZE) * 1.3
    max_pos_change = 0.1

    axes[0].plot(actual.t, x_error, label="Error")
    axes[0].set_title("x")
    axes[0].plot(actual.t, np.zeros_like(actual.t), linestyle="--", label="0")
    axes[0].legend()
    axes[0].set_ylim(-max_pos_change, max_pos_change)

    axes[1].plot(actual.t, y_error, label="Error")
    axes[1].plot(actual.t, np.zeros_like(actual.t), linestyle="--", label="0")
    axes[1].set_title("y")
    axes[1].legend()
    axes[1].set_ylim(-max_pos_change, max_pos_change)

    theta_error = actual.theta - desired.theta
    th_change = max(np.ptp(theta_error), MIN_AXIS_SIZE) * 2.0  # type: ignore
    th_change = 0.1

    axes[2].plot(actual.t, theta_error, label="Error")
    axes[2].set_title("theta")
    axes[2].plot(actual.t, np.zeros_like(actual.t), linestyle="--", label="0")
    axes[2].legend()
    axes[2].set_ylim(-th_change, th_change)

    if plot_lam:
        lam_error = actual.lam - desired.lam
        axes[3].plot(actual.t, lam_error, label="Error")
        axes[3].plot(actual.t, np.zeros_like(actual.t), linestyle="--", label="0")
        axes[3].set_title("lam")
        axes[3].legend()
        axes[3].set_ylim(0, 1)

    plt.tight_layout()
    file_name = f"planar_pushing_states_error{suffix}.pdf"
    file_path = f"{save_dir}/{file_name}" if save_dir else file_name
    plt.savefig(file_path)

    if not plot_control:
        return

    # Control input
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 8))

    if len(actual.c_n) == len(actual.t):
        pass
    elif len(actual.c_n) == len(actual.t) - 1:
        actual.t = actual.t[:-1]
    else:
        raise ValueError("Mismatch in data length")

    max_force_change = max(max(np.ptp(actual.c_n), np.ptp(actual.c_f)), MIN_AXIS_SIZE) * 2  # type: ignore

    # Plot lines on each subplot
    axes[0].plot(actual.t, actual.c_n, label="Actual")
    axes[0].plot(actual.t, desired.c_n, linestyle="--", label="Desired")
    axes[0].set_title("c_n")
    axes[0].legend()
    # axes[0].set_ylim(-max_force_change, max_force_change)

    axes[1].plot(actual.t, actual.c_f, label="Actual")
    axes[1].plot(actual.t, desired.c_f, linestyle="--", label="Desired")
    axes[1].set_title("c_f")
    axes[1].legend()
    # axes[1].set_ylim(-max_force_change, max_force_change)

    max_lam_dot_change = max(np.ptp(actual.lam_dot), MIN_AXIS_SIZE) * 1.3  # type: ignore
    axes[2].plot(actual.t, actual.lam_dot, label="Actual")
    axes[2].plot(actual.t, desired.lam_dot, linestyle="--", label="Desired")
    axes[2].set_title("lam_dot")
    axes[2].legend()
    # axes[2].set_ylim(-max_lam_dot_change, max_lam_dot_change)

    # Adjust layout
    plt.tight_layout()
    file_name = f"planar_pushing_control{suffix}.pdf"
    file_path = f"{save_dir}/{file_name}" if save_dir else file_name
    plt.savefig(file_path)


def plot_realtime_rate(
    real_time_rate_log: List[float],
    time_step: float,
    suffix: str = "",
    save_dir: Optional[str] = None,
) -> None:
    plt.figure()
    plt.plot(real_time_rate_log)
    plt.title("Realtime rate vs. timestep")
    plt.xticks(np.arange(0, len(real_time_rate_log), 1 / time_step), rotation=90)
    plt.xlabel("timestep")
    plt.ylabel("Realtime rate")
    # Add grid
    plt.grid()
    plt.tight_layout()
    file_name = f"planar_pushing_realtime_rate{suffix}.pdf"
    file_path = f"{save_dir}/{file_name}" if save_dir else file_name
    plt.savefig(file_path)

def plot_mpc_solve_times(
        solve_times_log: Dict[str, List[float]],
        suffix: str = "",
        save_dir: Optional[str] = None,
) -> None:
    fig, ax = plt.subplots()
    for key, solve_times in solve_times_log.items():
        ax.plot(solve_times, label=key)
    ax.set_title("MPC solve times vs. timestep")
    ax.set_xlabel("timestep")
    ax.set_ylabel("Solve times")
    ax.legend()
    ax.grid()
    plt.tight_layout()
    file_name = f"planar_pushing_mpc_solve_times{suffix}.png"
    file_path = f"{save_dir}/{file_name}" if save_dir else file_name
    plt.savefig(file_path)


def plot_joint_state_logs(joint_state_log, num_positions, suffix="", save_dir=""):
    num_velocities = joint_state_log.data().shape[0] - num_positions
    # Split the data into positions and velocities
    data = joint_state_log.data()
    sample_times = joint_state_log.sample_times()
    positions = data[:num_positions, :]
    velocities = data[num_positions:, :]

    # Create a figure with two subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # Plotting positions
    for i in range(num_positions):
        axs[0].plot(sample_times, positions[i, :], label=f"Joint {i+1}")
    axs[0].set_title("Joint Positions")
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("Position")
    axs[0].legend()

    # Plotting velocities
    for i in range(num_velocities):
        axs[1].plot(sample_times, velocities[i, :], label=f"Joint {i+1}")
    axs[1].set_title("Joint Velocities")
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Velocity")
    axs[1].legend()

    # Adjust layout
    plt.tight_layout()
    file_name = f"joint_states{suffix}.pdf"
    file_path = f"{save_dir}/{file_name}" if save_dir else file_name
    plt.savefig(file_path)


PLOT_WIDTH_INCH = 7
PLOT_HEIGHT_INCH = 4.5

DISTANCE_REF = 0.3  # Width of box
FORCE_REF = 10  # Current max force
TORQUE_REF = FORCE_REF * DISTANCE_REF  # Almost max torque


def plot_cos_sine_trajs(
    rot_trajs: npt.NDArray[np.float64],
    A: Optional[npt.NDArray[np.float64]] = None,
    b: Optional[npt.NDArray[np.float64]] = None,
    filename: Optional[str] = None,
):  # (num_steps, 2)
    """
    @param rot_trajs: Matrix of size (num_steps, 2), where the first col is
    cosines and the second col is sines.
    """
    # For some reason pyright complains about the typing being wrong with ax
    fig, ax = plt.subplots(1, 1)

    # Plot unit circle
    t = np.linspace(0, np.pi * 2, 100)
    ax.plot(np.cos(t), np.sin(t), color="grey", alpha=0.5)  # type: ignore

    ax.plot(  # type: ignore
        rot_trajs[:, 0],
        rot_trajs[:, 1],
        linestyle="--",
        marker="o",
        label="Ctrl points",
    )

    OFFSET = 0.1
    for i in range(rot_trajs.shape[0]):
        ax.annotate(str(i), (rot_trajs[i, 0] + OFFSET, rot_trajs[i, 1]))  # type: ignore

    # Plot start and end
    plt.plot(
        rot_trajs[0, 0], rot_trajs[0, 1], "go", label="start"
    )  # 'ro' indicates red color and circle marker
    plt.plot(rot_trajs[-1, 0], rot_trajs[-1, 1], "ro", label="target")

    ax.set_aspect("equal", "box")  # type: ignore
    ax.set_title("Effect of relaxing SO(2) constraints")  # type: ignore
    ax.set_xlabel(r"$\cos{\theta}$")  # type: ignore
    ax.set_ylabel(r"$\sin{\theta}$")  # type: ignore
    ax.legend(loc="lower left")  # type: ignore

    if A is not None and b is not None:
        x_values = np.linspace(-1.5, 1.5, 400)

        # Plot each line
        for i in range(A.shape[0]):
            if A[i, 1] != 0:
                # Solve for y
                y_values = (b[i] - A[i, 0] * x_values) / A[i, 1]
            else:
                # If A[i, 1] is zero, it's a vertical line
                x_values_line = np.full_like(x_values, b[i] / A[i, 0])
                y_values = x_values
                ax.plot(x_values_line, y_values, label=f"Line {i+1}")
                continue

            ax.plot(x_values, y_values, label=f"Line {i+1}")

    if filename is not None:
        fig.savefig(filename + "_rotations.png")  # type: ignore
    else:
        plt.show()


def show_plots() -> None:
    plt.show()


def _create_curve_norm(
    curve: npt.NDArray[np.float64],  # (N, dims)
) -> npt.NDArray[np.float64]:  # (N, 1)
    return np.apply_along_axis(np.linalg.norm, 1, curve).reshape((-1, 1))


def create_static_equilibrium_analysis(
    fb_violation_traj: npt.NDArray[np.float64],  # (N, dims)
    tb_violation_traj: npt.NDArray[np.float64],  # (N, 1)
    num_ctrl_points: int,
    mass: Optional[float] = None,  # reference values used for scaling
    width: Optional[float] = None,
):
    if mass is None or width is None:
        force_ref = 1
        torque_ref = 1
    else:
        force_ref = mass * 9.81
        torque_ref = force_ref * width / 2

    fb_norm_violation = _create_curve_norm(fb_violation_traj)

    N = fb_violation_traj.shape[0]
    x_axis = np.linspace(0, num_ctrl_points, N)

    fig, axs = plt.subplots(2, sharex=True)
    fig.suptitle("Force and torque balance violation")

    axs[0].plot(x_axis, fb_norm_violation)
    axs[0].set_title("Norm of force balance violation")
    axs[0].set(ylabel="[N]")
    axs[0].set_ylim(-force_ref, force_ref)

    axs[1].plot(x_axis, tb_violation_traj)
    axs[1].set_title("Torque balance violation")
    axs[1].set(xlabel="Time [s]", ylabel="[Nm]")
    axs[1].xaxis.set_ticks(np.arange(0, num_ctrl_points + 1))
    axs[1].set_ylim(-torque_ref, torque_ref)

    for ax in axs:
        ax.grid()

    fig.set_size_inches(PLOT_WIDTH_INCH, PLOT_HEIGHT_INCH)  # type: ignore
    fig.tight_layout()  # type: ignore


def create_forces_eq_and_opposite_analysis(
    sum_of_forces_traj: npt.NDArray[np.float64],  # (N, dims)
    num_ctrl_points: int,
    mass: Optional[float] = None,  # reference values used for scaling
    width: Optional[float] = None,
):
    if mass is None or width is None:
        force_ref = 1
        torque_ref = 1
    else:
        force_ref = mass * 9.81
        torque_ref = force_ref * width / 2

    N = sum_of_forces_traj.shape[0]
    x_axis = np.linspace(0, num_ctrl_points, N)

    fig, axs = plt.subplots(2, sharex=True)
    fig.suptitle("Violation of Newton's Third Law")

    axs[0].plot(x_axis, sum_of_forces_traj[:, 0])
    axs[0].set_title("Violation in x-direction")
    axs[0].set(ylabel="[N]")
    axs[0].set_ylim(-force_ref, force_ref)

    axs[1].plot(x_axis, sum_of_forces_traj[:, 1])
    axs[1].set_title("Violation in y-direction")
    axs[1].set(xlabel="Time [s]", ylabel="[N]")
    axs[1].xaxis.set_ticks(np.arange(0, num_ctrl_points + 1))
    axs[1].set_ylim(-torque_ref, torque_ref)

    for ax in axs:
        ax.grid()

    fig.set_size_inches(PLOT_WIDTH_INCH, PLOT_HEIGHT_INCH)  # type: ignore
    fig.tight_layout()  # type: ignore


def create_quasistatic_pushing_analysis(
    dynamics_violation: npt.NDArray[np.float64],  # (N, dims)
    num_ctrl_points: int,
    trans_velocity_ref: float,
    angular_velocity_ref: float,
):
    N = dynamics_violation.shape[0]
    x_axis = np.linspace(0, num_ctrl_points, N)

    trans_axis_max = trans_velocity_ref * 1.7
    ang_axis_max = angular_velocity_ref * 1.7

    fig, axs = plt.subplots(3, sharex=True)
    fig.suptitle("Violation of quasi-static dynamics with ellipsoidal limit surface")

    axs[0].plot(x_axis, dynamics_violation[:, 0])
    axs[0].set_title("Violation in $\dot x$")
    axs[0].set(ylabel="[m/s]")
    axs[0].set_ylim(-trans_axis_max, trans_axis_max)

    axs[1].plot(x_axis, dynamics_violation[:, 1])
    axs[1].set_title("Violation in $\dot y$")
    axs[1].set(xlabel="Time [s]", ylabel="[m/s]")
    axs[1].xaxis.set_ticks(np.arange(0, num_ctrl_points + 1))
    axs[1].set_ylim(-trans_axis_max, trans_axis_max)

    axs[2].plot(x_axis, dynamics_violation[:, 2])
    axs[2].set_title("Violation in $\omega$")
    axs[2].set(xlabel="Time [s]", ylabel="[rad/s]")
    axs[2].xaxis.set_ticks(np.arange(0, num_ctrl_points + 1))
    axs[2].set_ylim(-ang_axis_max, ang_axis_max)

    for ax in axs:
        ax.grid()

    fig.set_size_inches(PLOT_WIDTH_INCH, PLOT_HEIGHT_INCH)  # type: ignore
    fig.tight_layout()  # type: ignore


# TODO: These are likely outdated and should be updated


def create_newtons_third_law_analysis(
    equal_contact_point_ctrl_points: List[ContactFrameConstraints],
    equal_rel_position_ctrl_points: List[ContactFrameConstraints],
    newtons_third_law_ctrl_points: List[ContactFrameConstraints],
):
    # Local helper functions
    def _extract_ctrl_points_as_np(
        constraints: List[ContactFrameConstraints],
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        return np.hstack([c.in_frame_A for c in constraints]), np.hstack(
            [c.in_frame_B for c in constraints]
        )

    def _create_2d_curves(
        constraints: List[ContactFrameConstraints],
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        ctrl_points = _extract_ctrl_points_as_np(constraints)
        curves = tuple(
            BezierCurve.create_from_ctrl_points(cp).eval_entire_interval()
            for cp in ctrl_points
        )
        return curves

    def _create_2d_curve_norms(
        constraints: List[ContactFrameConstraints],
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        curves = _create_2d_curves(constraints)
        norm_curves = tuple(_create_curve_norm(c) for c in curves)
        return norm_curves

    eq_contact_point_A, eq_contact_point_B = _create_2d_curve_norms(
        equal_contact_point_ctrl_points
    )
    newtons_third_law_A, newtons_third_law_B = _create_2d_curve_norms(
        newtons_third_law_ctrl_points
    )
    eq_rel_position_A, eq_rel_position_B = _create_2d_curve_norms(
        equal_rel_position_ctrl_points
    )

    num_ctrl_points = len(equal_contact_point_ctrl_points)
    N = eq_contact_point_A.shape[0]
    x_axis = np.linspace(0, num_ctrl_points, N)

    fig, axs = plt.subplots(3, 2, sharey="row", sharex="col")  # type: ignore

    fig.suptitle("Newtons Third Law constraint violations")

    all_pos_curves = np.concatenate(
        (
            eq_contact_point_A,
            eq_contact_point_B,
            eq_rel_position_A,
            eq_rel_position_B,
        )
    )

    MIN_Y_AXIS_METER = min(min(all_pos_curves), -DISTANCE_REF) * 1.25
    MAX_Y_AXIS_METER = max(max(all_pos_curves), DISTANCE_REF) * 1.25

    axs[0, 0].plot(x_axis, eq_contact_point_A)
    axs[0, 1].plot(x_axis, eq_contact_point_B)
    axs[0, 0].set_title("Contact points: T frame")
    axs[0, 1].set_title("Contact points: B frame")
    axs[0, 0].set(ylabel="[m]")
    axs[0, 0].set_ylim(MIN_Y_AXIS_METER, MAX_Y_AXIS_METER)

    axs[1, 0].plot(x_axis, eq_rel_position_A)
    axs[1, 1].plot(x_axis, eq_rel_position_B)
    axs[1, 0].set_title("Relative positions: T frame")
    axs[1, 1].set_title("Relative positions: B frame")
    axs[1, 0].set(ylabel="[m]")
    axs[1, 0].set_ylim(MIN_Y_AXIS_METER, MAX_Y_AXIS_METER)

    axs[2, 0].plot(x_axis, newtons_third_law_A)
    axs[2, 1].plot(x_axis, newtons_third_law_B)
    axs[2, 0].set_title("Contact forces: T frame")
    axs[2, 1].set_title("Contact forces: B frame")
    axs[2, 0].set(ylabel="[N]")
    axs[2, 0].set_ylim(-1, FORCE_REF)

    axs[2, 0].set(xlabel="Control point")
    axs[2, 1].set(xlabel="Control point")

    for row in axs:
        for ax in row:
            ax.grid()

    # Only show ctrl point numbers
    axs[0, 0].xaxis.set_ticks(np.arange(0, num_ctrl_points + 1))

    fig.tight_layout()  # type: ignore
    fig.set_size_inches(10, 9)  # type: ignore


def create_force_plot(
    force_ctrl_points: List[npt.NDArray[np.float64]], force_names: List[str]
) -> None:
    MIN_Y_AXIS_NEWTON = -1
    MAX_Y_AXIS_NEWTON = 10

    force_curves = [
        BezierCurve.create_from_ctrl_points(points).eval_entire_interval()
        for points in force_ctrl_points
    ]
    force_norms = [_create_curve_norm(curve) for curve in force_curves]

    # TODO: For now I will hardcode which forces to plot, but this shold be generalized
    fig, axs = plt.subplots(2, 1, sharey=True, sharex="col")  # type: ignore

    num_ctrl_points = force_ctrl_points[0].shape[1]
    N = force_curves[0].shape[0]
    x_axis = np.linspace(0, num_ctrl_points, N)

    fig.suptitle("Norm of Contact Forces")
    axs[0].set_title("table/box")
    axs[0].plot(x_axis, np.hstack((force_norms[0], force_norms[1])))
    axs[0].set(ylabel="[N]")
    axs[0].legend(["in B frame", "in W frame"])
    axs[0].set_ylim(MIN_Y_AXIS_NEWTON, MAX_Y_AXIS_NEWTON)

    axs[1].set_title("box/finger")
    axs[1].plot(x_axis, force_norms[2])
    axs[1].set(ylabel="[N]")
    axs[1].set(xlabel="Control point")
    axs[1].legend(["in B frame"])

    for ax in axs:
        ax.grid()


from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from planning_through_contact.tools.utils import evaluate_np_expressions_array
from planning_through_contact.visualize.analysis import plot_cos_sine_trajs


def plot_constraint_violation(
    data: Dict[str, npt.NDArray[np.float64]],
    ref_vals: Dict[str, float],
    show_abs: bool = True,
    filename: Optional[str] = None,
) -> None:
    # Preparing the plot
    num_groups = len(data)
    fig, axs = plt.subplots(1, num_groups)
    group_names = list(data.keys())
    max_bars = max(
        [len(data[key].flatten()) for key in group_names]
    )  # maximum number of bars in a group
    bar_width = 0.8 / max_bars  # width of each bar
    opacity = 0.8

    # Colors for each subplot
    colors = ["red", "blue", "green", "purple", "orange"]

    # Creating the bars
    for i, (ax, key) in enumerate(zip(axs, group_names)):
        values = data[key].flatten()
        ref = ref_vals[key]
        if show_abs:
            values = np.abs(values)

        bar_positions = np.arange(len(values)) * (1 / max_bars) + i - 0.4
        color = colors[i]
        ax.bar(bar_positions, values, bar_width, alpha=opacity, color=color)

        # adjusting the labels
        ax.set_xlabel(f"{key}", rotation=45)
        ax.set_xticks([])

        # Draw reference value
        if ref is not None:
            if show_abs:
                ref = np.abs(ref)
            ax.axhline(ref, color="red", linestyle="--", label="ref")

        # Only show legend on last plot
        if i == len(data) - 1:
            ax.legend()

    # Show the plot
    fig.suptitle("Quadratic constraint violations")
    plt.tight_layout()  # Adjust the layout

    if filename is not None:
        fig.savefig(filename + "_constraints.png")  # type: ignore
    else:
        plt.show()


def analyze_plan(
    path: PlanarPushingPath,
    filename: Optional[str] = None,
    rounded: bool = False,
) -> None:
    face_modes = [
        pair.mode for pair in path.pairs if isinstance(pair.mode, FaceContactMode)
    ]
    face_vertices = [
        pair.vertex for pair in path.pairs if isinstance(pair.mode, FaceContactMode)
    ]
    if rounded:
        assert path.rounded_result is not None
        result = path.rounded_result
        path_knot_points = path.get_rounded_vars()
    else:
        result = path.result
        path_knot_points = path.get_vars()

    keys = face_modes[0].constraints.keys()
    constraint_violations = {key: [] for key in keys}
    for key in keys:
        for mode, vertex in zip(face_modes, face_vertices):
            for constraints in mode.constraints[key]:
                if not isinstance(
                    constraints, type(np.array([]))
                ):  # only one constraint
                    if rounded:
                        violation = mode.eval_binding(constraints, result)
                    else:
                        violation = mode.eval_binding_with_vertex_vars(
                            constraints, vertex, result
                        )

                    constraint_violations[key].append(violation)
                else:
                    if rounded:
                        violations = [
                            mode.eval_binding(constraint, result)
                            for constraint in constraints
                        ]
                    else:
                        violations = [
                            mode.eval_binding_with_vertex_vars(
                                constraint, vertex, result
                            )
                            for constraint in constraints
                        ]

                    constraint_violations[key].append(violations)

    # NOTE: This is super hacky
    for key, item in constraint_violations.items():
        constraint_violations[key] = np.array(item)  # type: ignore

    num_knot_points_in_path = sum((pair.mode.num_knot_points for pair in path.pairs))
    MIN_REF_THETA_VEL = np.pi / 15
    ref_theta_vel = max(
        np.mean(
            np.concatenate(
                [
                    points.delta_omega_WBs
                    for points in path_knot_points
                    if isinstance(points, FaceContactVariables)
                ]
            )
        ),
        MIN_REF_THETA_VEL,
    )
    MIN_REF_VEL = 0.05  # m/s
    ref_vel = max(
        np.mean(
            np.concatenate(
                [
                    [np.linalg.norm(v_WB) for v_WB in points.v_WBs]
                    for points in path_knot_points
                    if isinstance(points, FaceContactVariables)
                ]
            )
        ),
        MIN_REF_VEL,
    )
    ref_vals = {
        "SO2": 1,
        "rotational_dynamics": ref_theta_vel,
        "translational_dynamics": ref_vel,
        "translational_dynamics_red": ref_vel,
    }
    plot_constraint_violation(constraint_violations, ref_vals, filename=filename)

    # (num_knot_points, 2): first col cosines, second col sines

    face_contact_vars = [
        knot_points
        for knot_points in path_knot_points
        if isinstance(knot_points, FaceContactVariables)
    ]
    rs = np.vstack(
        [R_WB[:, 0] for knot_points in face_contact_vars for R_WB in knot_points.R_WBs]
    )
    plot_cos_sine_trajs(rs, filename=filename)


def analyze_mode_result(
    mode: FaceContactMode,
    traj: PlanarPushingTrajectory,
    result: MathematicalProgramResult,
    rank_analysis: bool = True,
) -> None:
    if rank_analysis:
        Xs = mode.get_Xs()

        if len(Xs) == 1:
            X_sol = result.GetSolution(Xs[0])

            eigs, _ = np.linalg.eig(X_sol)
            norms = np.abs(eigs)

            plt.bar(range(len(norms)), norms)
            plt.xlabel("Index of Eigenvalue")
            plt.ylabel("Norm of Eigenvalue")
            plt.title("Norms of the Eigenvalues of the Matrix")
            plt.show()
        else:
            X_sols = [evaluate_np_expressions_array(X, result) for X in Xs]

            eigs, _ = zip(*[np.linalg.eig(X_sol) for X_sol in X_sols])
            norms = [np.abs(eig) for eig in eigs]

            data = [
                [norm[i] if i < len(norm) else 0 for norm in norms]
                for i in range(len(norms[0]))
            ]

            means = [np.mean(sublist) for sublist in data]
            std_devs = [np.std(sublist) for sublist in data]

            plt.bar(range(len(means)), means, yerr=std_devs)
            plt.xlabel("Index of Eigenvalue")
            plt.ylabel("Norm of Eigenvalue")
            plt.title("Norms of the Eigenvalues of the Matrix")
            plt.show()

    keys = mode.constraints.keys()
    constraint_violations = {key: [] for key in keys}
    for key in keys:
        for constraints in mode.constraints[key]:
            if not isinstance(constraints, type(np.array([]))):  # only one constraint
                constraint_violations[key].append(
                    mode.eval_binding(constraints, result)
                )
            else:
                constraint_violations[key].append(
                    [
                        mode.eval_binding(constraint, result)
                        for constraint in constraints
                    ]
                )

    # NOTE: This is super hacky
    for key, item in constraint_violations.items():
        constraint_violations[key] = np.array(item)

    ref_vals = {
        "SO2": 1,
        "rotational_dynamics": np.mean(traj.path_knot_points[0].delta_omega_WBs),
        "translational_dynamics": np.mean(traj.path_knot_points[0].v_WBs),
        "translational_dynamics_red": np.mean(traj.path_knot_points[0].v_WBs),
    }
    if traj.path_knot_points[0].theta_dots is not None:
        ref_vals["exponential_map"] = np.mean(traj.path_knot_points[0].theta_dots)
    plot_constraint_violation(constraint_violations, ref_vals)

    # (num_knot_points, 2): first col cosines, second col sines

    rs = np.vstack([R_WB[:, 0] for R_WB in traj.path_knot_points[0].R_WBs])
    plot_cos_sine_trajs(rs)
