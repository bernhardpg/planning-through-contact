from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pydrake.geometry.optimization as opt
from pydrake.solvers import MathematicalProgramResult
from pydrake.systems.primitives import VectorLog

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
        # Padding state since we didn't log lam
        PAD_VAL = 0
        state_np_array = np.vstack(
            (state_np_array, PAD_VAL * np.ones_like(state_np_array[0, :]))
        )
        control_np_array = np.ones((3, len(t))) * PAD_VAL
        return cls.from_np(t, state_np_array, control_np_array)


def plot_planar_pushing_logs(
    state_log: VectorLog,
    state_log_desired: VectorLog,
    control_log: VectorLog,
    control_log_desired: VectorLog,
) -> None:
    actual = PlanarPushingLog.from_log(state_log, control_log)
    desired = PlanarPushingLog.from_log(state_log_desired, control_log_desired)

    plot_planar_pushing_trajectory(actual, desired)


def plot_planar_pushing_logs_from_pose_vectors(
    pusher_pose_vector_log: VectorLog,
    slider_pose_vector_log: VectorLog,
    pusher_pose_vector_log_desired: VectorLog,
    slider_pose_vector_log_desired: VectorLog,
) -> None:
    actual_pusher = PlanarPushingLog.from_pose_vector_log(pusher_pose_vector_log)
    actual_slider = PlanarPushingLog.from_pose_vector_log(slider_pose_vector_log)
    desired_pusher = PlanarPushingLog.from_pose_vector_log(
        pusher_pose_vector_log_desired
    )
    desired_slider = PlanarPushingLog.from_pose_vector_log(
        slider_pose_vector_log_desired
    )
    plot_planar_pushing_trajectory(actual_slider, desired_slider, suffix="_slider")
    plot_planar_pushing_trajectory(actual_pusher, desired_pusher, suffix="_pusher")


def plot_planar_pushing_trajectory(
    actual: PlanarPushingLog, desired: PlanarPushingLog, suffix: str = ""
) -> None:
    # State plot
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(8, 8))
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

    axes[3].plot(actual.t, actual.lam, label="Actual")
    axes[3].plot(actual.t, desired.lam, linestyle="--", label="Desired")
    axes[3].set_title("lam")
    axes[3].legend()
    axes[3].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(f"planar_pushing_states{suffix}.pdf")

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
    axes[0].set_ylim(-max_force_change, max_force_change)

    axes[1].plot(actual.t, actual.c_f, label="Actual")
    axes[1].plot(actual.t, desired.c_f, linestyle="--", label="Desired")
    axes[1].set_title("c_f")
    axes[1].legend()
    axes[1].set_ylim(-max_force_change, max_force_change)

    max_lam_dot_change = max(np.ptp(actual.lam_dot), MIN_AXIS_SIZE) * 1.3  # type: ignore
    axes[2].plot(actual.t, actual.lam_dot, label="Actual")
    axes[2].plot(actual.t, desired.lam_dot, linestyle="--", label="Desired")
    axes[2].set_title("lam_dot")
    axes[2].legend()
    axes[2].set_ylim(-max_lam_dot_change, max_lam_dot_change)

    # Adjust layout
    plt.tight_layout()
    plt.savefig(f"planar_pushing_control{suffix}.pdf")


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
    path: PlanarPushingPath, traj: PlanarPushingTrajectory, filename: str
) -> None:
    face_modes = [
        pair.mode for pair in path.pairs if isinstance(pair.mode, FaceContactMode)
    ]
    face_vertices = [
        pair.vertex for pair in path.pairs if isinstance(pair.mode, FaceContactMode)
    ]
    result = path.result

    keys = face_modes[0].constraints.keys()
    constraint_violations = {key: [] for key in keys}
    for key in keys:
        for mode, vertex in zip(face_modes, face_vertices):
            for constraints in mode.constraints[key]:
                if not isinstance(
                    constraints, type(np.array([]))
                ):  # only one constraint
                    constraint_violations[key].append(
                        mode.eval_binding_with_vertex_vars(constraints, vertex, result)
                    )
                else:
                    constraint_violations[key].append(
                        [
                            mode.eval_binding_with_vertex_vars(
                                constraint, vertex, result
                            )
                            for constraint in constraints
                        ]
                    )

    # NOTE: This is super hacky
    for key, item in constraint_violations.items():
        constraint_violations[key] = np.array(item)

    num_knot_points_in_path = sum((pair.mode.num_knot_points for pair in path.pairs))
    ref_theta_vel = np.mean(
        np.concatenate(
            [
                points.delta_omega_WBs
                for points in traj.path_knot_points
                if isinstance(points, FaceContactVariables)
            ]
        )
    )
    ref_vel = np.mean(
        np.concatenate(
            [
                [np.linalg.norm(v_WB) for v_WB in points.v_WBs]
                for points in traj.path_knot_points
                if isinstance(points, FaceContactVariables)
            ]
        )
    )
    ref_vals = {
        "SO2": 1,
        "rotational_dynamics": ref_theta_vel,
        "translational_dynamics": ref_vel,
        "translational_dynamics_red": ref_vel,
    }
    plot_constraint_violation(constraint_violations, ref_vals, filename=filename)

    # (num_knot_points, 2): first col cosines, second col sines

    rs = np.vstack(
        [
            R_WB[:, 0]
            for idx in range(len(traj.path_knot_points))
            for R_WB in traj.path_knot_points[idx].R_WBs
        ]
    )
    plot_cos_sine_trajs(rs, filename=filename)


def analyze_mode_result(
    mode: FaceContactMode,
    traj: PlanarPushingTrajectory,
    result: MathematicalProgramResult,
) -> None:
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

    constraint_violations = {
        key: evaluate_np_expressions_array(value, result)
        for key, value in mode.constraints.items()
    }

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
