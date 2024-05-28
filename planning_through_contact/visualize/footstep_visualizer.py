from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Ellipse, FancyArrowPatch, Polygon

from planning_through_contact.planning.footstep.footstep_plan_config import PotatoRobot
from planning_through_contact.planning.footstep.footstep_trajectory import FootstepPlan
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

    def animate(step: int) -> None:
        time = step * plan.dt

        # Robot position and orientation
        p_WB_val = plan.get(time, "p_WB").flatten()  # type: ignore
        theta_WB_val = plan.get(time, "theta_WB")
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

    # Create and display animation
    n_steps = plan.num_knot_points + 1
    ani = FuncAnimation(fig, animate, frames=n_steps, interval=plan.dt * 1000)  # type: ignore
    if output_file is not None:
        ani.save(f"{output_file}.mp4", writer="ffmpeg")

    return ani


def plot_relaxation_errors(
    plan: FootstepPlan,
    title: Optional[str] = None,
    output_file: Optional[str] = None,
) -> None:
    # Assuming compute_torques method is defined in FootstepPlan
    planned_torques = plan.tau_F_Ws  # List of Lists of np.ndarray
    true_torques = plan.compute_torques()  # List of Lists of np.ndarray

    num_feet = len(planned_torques)
    num_forces = len(planned_torques[0]) if num_feet > 0 else 0

    # Determine global y-axis limits
    all_torques = [torque for sublist in planned_torques for torque in sublist] + [
        torque for sublist in true_torques for torque in sublist
    ]
    y_min = min(torque.min() for torque in all_torques)
    y_max = max(torque.max() for torque in all_torques)

    fig, axs = plt.subplots(
        num_feet, num_forces, figsize=(12, 3.5 * num_feet), squeeze=False
    )

    for i in range(num_feet):
        for j in range(num_forces):
            ax = axs[i, j]
            planned_torque = planned_torques[i][j]
            true_torque = true_torques[i][j]
            N = planned_torque.shape[0]
            x = np.arange(N)

            ax.plot(x, planned_torque, label="Planned Torque")
            ax.plot(x, true_torque, label="True Torque", linestyle="--")
            ax.set_xlabel("N")
            ax.set_ylabel("Torque")
            ax.set_title(f"Foot {i + 1} - Force {j + 1}")
            ax.legend()
            ax.set_ylim(y_min, y_max)

    if title:
        fig.suptitle(title)
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file)
        plt.close()
