from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Ellipse, FancyArrowPatch, Polygon
from matplotlib.ticker import MaxNLocator

from planning_through_contact.geometry.utilities import cross_2d
from planning_through_contact.planning.footstep.footstep_plan_config import PotatoRobot
from planning_through_contact.planning.footstep.footstep_trajectory import (
    FootstepPlan,
    FootstepPlanResult,
)
from planning_through_contact.planning.footstep.in_plane_terrain import InPlaneTerrain


def animate_footstep_plan(
    robot: PotatoRobot,
    terrain: InPlaneTerrain,
    plan: FootstepPlan,
    title: Optional[str] = None,
    output_file: Optional[str] = None,
) -> FuncAnimation:
    # Initialize figure for animation
    fig, ax = plt.subplots()

    # Plot stepping stones
    terrain.plot(title=title, ax=ax, max_height=2.5)

    # Plot robot
    robot_body = Ellipse(
        xy=(0, 0),
        width=robot.width,
        height=robot.height,
        angle=0,
        edgecolor="black",
        facecolor="none",
    )
    ax.add_patch(robot_body)

    # Foot
    base_foot_vertices = np.array(
        [
            [-robot.foot_length / 2, 0],
            [robot.foot_length / 2, 0],
            [0, robot.foot_height],
        ]
    )
    NUM_FEET = 2
    feet = [
        Polygon(base_foot_vertices, closed=True, fill="blue", edgecolor="black")
        for _ in range(NUM_FEET)
    ]
    for foot in feet:
        ax.add_patch(foot)

    # Forces
    FORCE_SCALE = 1e-3

    def _create_force_patch():
        force = FancyArrowPatch(
            posA=(0, 0),
            posB=(1 * FORCE_SCALE, 1 * FORCE_SCALE),
            arrowstyle="->",
            color="green",
        )
        return force

    NUM_FORCES_PER_FOOT = 2
    foot_forces = [  # forces for each contact point on each foot
        [_create_force_patch() for _ in range(NUM_FORCES_PER_FOOT)]
        for _ in range(NUM_FEET)
    ]
    for f1, f2 in foot_forces:
        ax.add_patch(f1)
        ax.add_patch(f2)

    p_WB = ax.scatter(0, 0, color="r", zorder=3, label="CoM")

    # Misc settings
    plt.close()
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.3), ncol=2)

    # Create and display animation
    end_time = plan.end_time
    animation_dt_ms = 0.01 * 1000
    num_frames = int(np.ceil(end_time * 1000 / animation_dt_ms))

    def animate(step: int) -> None:
        time = step * animation_dt_ms / 1000

        # Robot position and orientation
        p_WB_val = plan.get(time, "p_WB").flatten()  # type: ignore
        theta_WB_val = plan.get(time, "theta_WB")
        assert type(theta_WB_val) is float
        if not np.isnan(p_WB_val).any():
            p_WB.set_offsets(p_WB_val)
            robot_body.set_center(p_WB_val)  # type: ignore
            robot_body.angle = theta_WB_val * 180 / np.pi
            p_WB.set_visible(True)
            robot_body.set_visible(True)
        else:
            p_WB.set_visible(False)
            robot_body.set_visible(False)

        # Feet position
        for foot_idx in range(NUM_FEET):
            p_WF_val = plan.get_foot(foot_idx, time, "p_WF").flatten()  # type: ignore
            if not np.isnan(p_WF_val).any():
                feet[foot_idx].set_xy(base_foot_vertices + p_WF_val)
                feet[foot_idx].set_visible(True)
            else:
                feet[foot_idx].set_visible(False)

        # Feet forces
        for foot_idx in range(NUM_FEET):
            f_F_Ws_val = plan.get_foot(foot_idx, time, "f_F_Ws")
            p_WFcs_val = plan.get_foot(foot_idx, time, "p_WFcs")

            forces = foot_forces[foot_idx]

            for force_idx, (f, p) in enumerate(zip(f_F_Ws_val, p_WFcs_val)):
                f = f.flatten()
                p = p.flatten()
                if not np.isnan(f).any():
                    forces[force_idx].set_positions(posA=p, posB=(p + f * FORCE_SCALE))
                    forces[force_idx].set_visible(True)
                else:
                    forces[force_idx].set_visible(False)

    ani = FuncAnimation(fig, animate, frames=num_frames, interval=animation_dt_ms)  # type: ignore
    if output_file is not None:
        if "mp4" in output_file:
            output_file = output_file.split(".")[0]
        ani.save(f"{output_file}.mp4", writer="ffmpeg")

    return ani


def plot_relaxation_vs_rounding_bar_plot(
    plan_results: List[FootstepPlanResult],
    filename: Optional[str] = None,
    best_res: Optional[str] = None,
) -> None:
    # Plot histogram over costs
    for res in plan_results:
        assert res.gcs_metrics is not None

    gcs_times = [res.gcs_metrics.solve_time for res in plan_results]  # type: ignore
    gcs_costs = [res.gcs_metrics.cost for res in plan_results]  # type: ignore
    relaxed_costs = [res.restriction_metrics.cost for res in plan_results]
    relaxed_times = [res.restriction_metrics.solve_time for res in plan_results]

    rounded_costs = [res.rounded_metrics.cost for res in plan_results]
    rounded_times = [res.rounded_metrics.solve_time for res in plan_results]

    fig, axs = plt.subplots(2, 1, figsize=(6, 8))

    unique_names = [res.get_unique_gcs_name() for res in plan_results]

    def _plot_bar_plot_pair(ax, categories, values, best_res):

        # Number of categories
        n = len(values[0])

        # Positions of the bars on the x-axis
        ind = np.arange(n)

        num_categories = len(categories)
        # Width of a bar
        width = 0.8 / num_categories  # Total width divided by number of categories

        # Plotting the bars
        bars = []
        for i, (category, value) in enumerate(zip(categories, values)):
            # Calculate the position for each category's bars
            position = ind - (0.8 - width) / 2 + i * width
            bars.append(ax.bar(position, value, width, label=category))

        # Adding labels and titles
        ax.legend()

        # Set x-ticks to match the number of bars
        ax.set_xticks(ind)
        ax.set_xticklabels(unique_names, rotation=45, ha="right")

        # Highlight the best index label
        if best_res is not None:
            xticklabels = ax.get_xticklabels()
            for label in xticklabels:
                if label.get_text() == best_res:
                    label.set_color("green")
                    label.set_fontweight("bold")

        # Add text annotations with height of bars for the categories
        for category in categories:
            cat_idx = categories.index(category)
            fontsize = 5  # matplotlib default
            for bar in bars[cat_idx]:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f"{height:.2f}",
                    ha="center",
                    va="bottom",
                    rotation=45,
                    color=bar.get_facecolor(),
                    fontsize=fontsize,
                )

    axs[0].set_title("Costs")
    _plot_bar_plot_pair(
        axs[0],
        ["gcs", "relaxed", "rounded"],
        [gcs_costs, relaxed_costs, rounded_costs],
        best_res,
    )

    axs[1].set_title("Solve times")
    axs[1].set_ylabel("Time [s]")
    _plot_bar_plot_pair(
        axs[1],
        ["gcs", "relaxed", "rounded"],
        [gcs_times, relaxed_times, rounded_times],
        best_res,
    )

    fig.tight_layout()

    if filename is not None:
        fig.savefig(filename.split(".")[0] + ".pdf")

    plt.close()


def visualize_feet_trajectories(
    plan: FootstepPlan, filename: Optional[str] = None
) -> None:
    DT = 1e-3
    times = np.arange(0, plan.end_time, DT)
    # the last dt is unused and we have N-1 knot points for inputs
    knot_point_times = np.concatenate([[0], np.cumsum(plan.dts)[:-2]])
    num_feet = plan.num_feet
    TWO_D = 2
    num_trajs_per_foot_contact_point = 2 * TWO_D + 1  # force and position + torque
    num_contact_points_per_foot = 2

    fig, axs = plt.subplots(
        num_trajs_per_foot_contact_point,
        num_feet * num_contact_points_per_foot,
        figsize=(12, 7),
        sharex=True,
    )

    for foot_idx in range(num_feet):
        for contact_point_idx in range(2):
            plot_idx = foot_idx * 2 + contact_point_idx
            ax_force_x = axs[0, plot_idx]
            ax_force_y = axs[1, plot_idx]
            ax_pos_x = axs[2, plot_idx]
            ax_pos_y = axs[3, plot_idx]
            ax_torque = axs[4, plot_idx]

            f_F_Ws_for_foot = []
            p_BFc_Ws_for_foot = []
            computed_tau_F_Ws_for_foot = []
            planned_tau_F_Ws_for_foot = []

            for t in times:
                f_F_Ws = plan.get_foot(foot_idx, t, "f_F_Ws")[contact_point_idx]  # type: ignore
                p_BFc_Ws = plan.get_foot(foot_idx, t, "p_BFc_Ws")[contact_point_idx]  # type: ignore
                computed_tau_F_Ws = plan.get_foot(foot_idx, t, "computed_tau_F_Ws")[contact_point_idx]  # type: ignore
                planned_tau_F_Ws = plan.get_foot(foot_idx, t, "planned_tau_F_Ws")[contact_point_idx]  # type: ignore

                f_F_Ws_for_foot.append(f_F_Ws)
                p_BFc_Ws_for_foot.append(p_BFc_Ws)
                computed_tau_F_Ws_for_foot.append(computed_tau_F_Ws)
                planned_tau_F_Ws_for_foot.append(planned_tau_F_Ws)

            f_F_Ws_for_foot = np.hstack(f_F_Ws_for_foot)  # (2, N)
            p_BFc_Ws_for_foot = np.hstack(p_BFc_Ws_for_foot)  # (2, N)
            computed_tau_F_Ws_for_foot = np.array(computed_tau_F_Ws_for_foot)
            planned_tau_F_Ws_for_foot = np.array(planned_tau_F_Ws_for_foot)

            ax_force_x.set_title(f"Foot {foot_idx}, cp {contact_point_idx}")
            ax_force_x.plot(times, f_F_Ws_for_foot[0, :], label="f_F_W_x")
            ax_force_y.plot(times, f_F_Ws_for_foot[1, :], label="f_F_W_y")
            ax_pos_x.plot(times, p_BFc_Ws_for_foot[0, :], label="p_BF_W_x")
            ax_pos_y.plot(times, p_BFc_Ws_for_foot[1, :], label="p_BF_W_y")
            ax_torque.plot(times, planned_tau_F_Ws_for_foot, label="tau_F_Ws (planned)")
            ax_torque.plot(
                times, computed_tau_F_Ws_for_foot, label="tau_F_Ws (computed)"
            )

            # Adding scatter plots at knot points without labels
            f_F_Ws_at_knots = np.array(
                [
                    plan.get_foot(foot_idx, t, "f_F_Ws")[contact_point_idx]  # type: ignore
                    for t in knot_point_times
                ]
            )
            p_BFc_Ws_at_knots = np.array(
                [
                    plan.get_foot(foot_idx, t, "p_BFc_Ws")[contact_point_idx]  # type: ignore
                    for t in knot_point_times
                ]
            )
            computed_tau_F_Ws_at_knots = np.array(
                [
                    plan.get_foot(foot_idx, t, "computed_tau_F_Ws")[  # type:ignore
                        contact_point_idx
                    ]
                    for t in knot_point_times
                ]
            )
            planned_tau_F_Ws_at_knots = np.array(
                [
                    plan.get_foot(foot_idx, t, "planned_tau_F_Ws")[contact_point_idx]  # type: ignore
                    for t in knot_point_times
                ]
            )

            ax_force_x.scatter(knot_point_times, f_F_Ws_at_knots[:, 0], color="r")
            ax_force_y.scatter(knot_point_times, f_F_Ws_at_knots[:, 1], color="r")
            ax_pos_x.scatter(knot_point_times, p_BFc_Ws_at_knots[:, 0], color="r")
            ax_pos_y.scatter(knot_point_times, p_BFc_Ws_at_knots[:, 1], color="r")
            ax_torque.scatter(knot_point_times, planned_tau_F_Ws_at_knots, color="r")
            ax_torque.scatter(knot_point_times, computed_tau_F_Ws_at_knots, color="b")

            ax_torque.set_xlabel("Time [s]")

    for ax in axs.flatten():
        ax.legend(fontsize="small", labelspacing=0.2, borderpad=0.3)

    plt.tight_layout()

    if filename is not None:
        fig.savefig(filename.split(".")[0] + ".pdf")


def visualize_footstep_plan_trajectories(
    robot: PotatoRobot,
    plan: FootstepPlan,
    title: Optional[str] = None,
    filename: Optional[str] = None,
) -> None:
    DT = 1e-3
    times = np.arange(0, plan.end_time, DT)
    # the last dt is unused and we have N-1 knot points for inputs
    knot_point_times = np.concatenate([[0], np.cumsum(plan.dts)[:-1]])

    p_WB_x = []
    p_WB_y = []
    theta_WB = []

    for t in times:
        p_WB = plan.get(t, "p_WB").flatten()  # type: ignore
        theta = plan.get(t, "theta_WB")

        if isinstance(p_WB, np.ndarray) and p_WB.shape == (2,):
            p_WB_x.append(p_WB[0])
            p_WB_y.append(p_WB[1])
        else:
            raise ValueError(
                f"Unexpected shape for p_WB at time {t}: {p_WB.shape if isinstance(p_WB, np.ndarray) else 'not an array'}"
            )

        if isinstance(theta, np.ndarray) and theta.shape == ():
            theta_WB.append(theta)
        elif isinstance(theta, float):
            theta_WB.append(theta)
        else:
            raise ValueError(
                f"Unexpected shape for theta_WB at time {t}: {theta.shape if isinstance(theta, np.ndarray) else 'not an array'}"
            )

    p_WB_x = np.array(p_WB_x)
    p_WB_y = np.array(p_WB_y)
    theta_WB = np.array(theta_WB)

    # Feet forces
    NUM_FEET = 2
    f_F_Ws_x_sum = []
    f_F_Ws_y_sum = []

    GRAV_FORCE = robot.mass * 9.81

    for t in times:
        f_F_Ws_at_t = [
            f
            for foot_idx in range(NUM_FEET)
            for f in plan.get_foot(foot_idx, t, "f_F_Ws")  # type: ignore
        ]
        sum_f_F_Ws_at_t = np.sum(f_F_Ws_at_t, axis=0)

        f_F_Ws_x_sum.append(sum_f_F_Ws_at_t[0])
        f_F_Ws_y_sum.append(sum_f_F_Ws_at_t[1] - GRAV_FORCE)

    f_F_Ws_x_sum = np.array(f_F_Ws_x_sum)
    f_F_Ws_y_sum = np.array(f_F_Ws_y_sum)

    planned_tau_F_Ws_sum = []
    for t in times:
        # First sum is over forces within one foot, second sum is over both feet
        tau_F_Ws_sum_at_t = np.sum(
            [
                np.sum(plan.get_foot(foot_idx, t, "planned_tau_F_Ws"))
                for foot_idx in range(NUM_FEET)
            ]
        )
        planned_tau_F_Ws_sum.append(tau_F_Ws_sum_at_t)

    planned_tau_F_Ws_sum = np.array(planned_tau_F_Ws_sum)

    actual_tau_F_Ws_sum = []
    for t in times:
        tau_F_Ws_sum_at_t = np.sum(
            [
                np.sum(plan.get_foot(foot_idx, t, "computed_tau_F_Ws"))
                for foot_idx in range(NUM_FEET)
            ]
        )
        actual_tau_F_Ws_sum.append(tau_F_Ws_sum_at_t)

    actual_tau_F_Ws_sum = np.array(actual_tau_F_Ws_sum)

    # Extract knot point values
    knot_p_WB_x = []
    knot_p_WB_y = []
    knot_theta_WB = []
    knot_f_F_Ws_x_sum = []
    knot_f_F_Ws_y_sum = []
    knot_planned_tau_F_Ws_sum = []
    knot_actual_tau_F_Ws_sum = []

    for t in knot_point_times:
        knot_p_WB = plan.get(t, "p_WB").flatten()  # type: ignore
        knot_theta = plan.get(t, "theta_WB")

        if isinstance(knot_p_WB, np.ndarray) and knot_p_WB.shape == (2,):
            knot_p_WB_x.append(knot_p_WB[0])
            knot_p_WB_y.append(knot_p_WB[1])
        else:
            raise ValueError(
                f"Unexpected shape for knot p_WB at time {t}: {knot_p_WB.shape if isinstance(knot_p_WB, np.ndarray) else 'not an array'}"
            )

        if isinstance(knot_theta, np.ndarray) and knot_theta.shape == ():
            knot_theta_WB.append(knot_theta)
        elif isinstance(knot_theta, float):
            knot_theta_WB.append(knot_theta)
        else:
            raise ValueError(
                f"Unexpected shape for knot theta_WB at time {t}: {knot_theta.shape if isinstance(knot_theta, np.ndarray) else 'not an array'}"
            )

        f_F_Ws_at_t = [
            f
            for foot_idx in range(NUM_FEET)
            for f in plan.get_foot(foot_idx, t, "f_F_Ws")  # type: ignore
        ]
        sum_f_F_Ws_at_t = np.sum(f_F_Ws_at_t, axis=0)
        knot_f_F_Ws_x_sum.append(sum_f_F_Ws_at_t[0])
        knot_f_F_Ws_y_sum.append(sum_f_F_Ws_at_t[1] - GRAV_FORCE)

        tau_F_Ws_sum_at_t = np.sum(
            [
                np.sum(plan.get_foot(foot_idx, t, "planned_tau_F_Ws"))
                for foot_idx in range(NUM_FEET)
            ]
        )
        knot_planned_tau_F_Ws_sum.append(tau_F_Ws_sum_at_t)

        tau_F_Ws_sum_at_t = np.sum(
            [
                np.sum(plan.get_foot(foot_idx, t, "computed_tau_F_Ws"))
                for foot_idx in range(NUM_FEET)
            ]
        )
        knot_actual_tau_F_Ws_sum.append(tau_F_Ws_sum_at_t)

    knot_p_WB_x = np.array(knot_p_WB_x)
    knot_p_WB_y = np.array(knot_p_WB_y)
    knot_theta_WB = np.array(knot_theta_WB)
    knot_f_F_Ws_x_sum = np.array(knot_f_F_Ws_x_sum[:-1])
    knot_f_F_Ws_y_sum = np.array(knot_f_F_Ws_y_sum[:-1])
    knot_planned_tau_F_Ws_sum = np.array(knot_planned_tau_F_Ws_sum[:-1])
    knot_actual_tau_F_Ws_sum = np.array(knot_actual_tau_F_Ws_sum[:-1])

    fig, axs = plt.subplots(6, 1, figsize=(6, 5), sharex=True)

    axs[0].plot(times, p_WB_x, label="p_WB x")
    axs[0].scatter(knot_point_times, knot_p_WB_x, color="r")
    axs[0].set_ylabel("[m]")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(times, p_WB_y, label="p_WB y")
    axs[1].scatter(knot_point_times, knot_p_WB_y, color="r")
    axs[1].set_ylabel("[m]")
    axs[1].legend()
    axs[1].grid(True)

    axs[2].plot(times, theta_WB * 180 / np.pi, label="theta_WB")
    axs[2].scatter(knot_point_times, knot_theta_WB * 180 / np.pi, color="r")
    axs[2].set_ylabel("[deg]")
    axs[2].legend()
    axs[2].grid(True)

    axs[3].plot(times, f_F_Ws_x_sum, label="sum(f_F_Ws)_x")
    axs[3].scatter(knot_point_times[:-1], knot_f_F_Ws_x_sum, color="r")
    axs[3].set_ylabel("[N]")
    axs[3].legend()
    axs[3].grid(True)

    axs[4].plot(times, f_F_Ws_y_sum, label="sum(f_F_Ws)_y")
    axs[4].scatter(knot_point_times[:-1], knot_f_F_Ws_y_sum, color="r")
    axs[4].set_ylabel("[N]")
    axs[4].legend()
    axs[4].grid(True)

    axs[5].plot(times, planned_tau_F_Ws_sum, label="sum(tau_F_Ws) (planned)")
    axs[5].plot(times, actual_tau_F_Ws_sum, label="sum(tau_F_Ws) (actual)")
    axs[5].scatter(knot_point_times[:-1], knot_planned_tau_F_Ws_sum, color="r")
    axs[5].scatter(knot_point_times[:-1], knot_actual_tau_F_Ws_sum, color="b")
    axs[5].set_ylabel("[Nm]")
    axs[5].legend()
    axs[5].grid(True)

    if title:
        fig.suptitle(title)

    axs[5].set_xlabel("Time (s)")

    plt.tight_layout()

    if filename is not None:
        fig.savefig(filename.split(".")[0] + ".pdf")

    plt.close()
