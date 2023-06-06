from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from geometry.bezier import BezierCurve
from geometry.two_d.contact.contact_pair_2d import ContactFrameConstraints

PLOT_WIDTH_INCH = 7
PLOT_HEIGHT_INCH = 4.5

DISTANCE_REF = 0.3  # Width of box
FORCE_REF = 10  # Current max force
TORQUE_REF = FORCE_REF * DISTANCE_REF  # Almost max torque


def plot_cos_sine_trajs(rot_trajs: npt.NDArray[np.float64]):  # (num_steps, 2)
    # For some reason pyright complains about the typing being wrong with ax
    _, ax = plt.subplots(1, 1)

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
    ax.set_aspect("equal", "box")  # type: ignore
    ax.set_title("Effect of relaxing SO(2) constraints")  # type: ignore
    ax.set_xlabel(r"$\cos{\theta}$")  # type: ignore
    ax.set_ylabel(r"$\sin{\theta}$")  # type: ignore
    ax.legend()  # type: ignore

    OFFSET = 0.1
    for i in range(rot_trajs.shape[0]):
        ax.annotate(str(i), (rot_trajs[i, 0] + OFFSET, rot_trajs[i, 1]))  # type: ignore

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
