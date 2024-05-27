from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Ellipse, FancyArrowPatch, Polygon

from planning_through_contact.planning.footstep.footstep_plan_config import PotatoRobot
from planning_through_contact.planning.footstep.footstep_trajectory import (
    FootstepTrajectory,
)
from planning_through_contact.planning.footstep.in_plane_terrain import InPlaneTerrain


def animate_footstep_plan(
    robot: PotatoRobot,
    terrain: InPlaneTerrain,
    plan: FootstepTrajectory,
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
    foot_left = Polygon(base_foot_vertices, closed=True, fill="blue", edgecolor="black")
    ax.add_patch(foot_left)
    foot_right = Polygon(
        base_foot_vertices, closed=True, fill="green", edgecolor="black"
    )
    ax.add_patch(foot_right)

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

    force_l1 = _create_force_patch()
    ax.add_patch(force_l1)
    force_l2 = _create_force_patch()
    ax.add_patch(force_l2)
    force_r1 = _create_force_patch()
    ax.add_patch(force_r1)
    force_r2 = _create_force_patch()
    ax.add_patch(force_r2)

    # Initial position of the feet
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

        # Left foot
        p_WF1_val = plan.get(time, "p_WF1").flatten()  # type: ignore
        if not np.isnan(p_WF1_val).any():
            foot_left.set_xy(base_foot_vertices + p_WF1_val)
            foot_left.set_visible(True)
        else:
            foot_left.set_visible(False)

        # Right foot
        p_WF2_val = plan.get(time, "p_WF2").flatten()  # type: ignore
        if p_WF2_val is not None and not np.isnan(p_WF2_val).any():
            foot_right.set_xy(base_foot_vertices + p_WF2_val)
            foot_right.set_visible(True)
        else:
            foot_right.set_visible(False)

        # Forces for left foot
        f_F1_1W_val = plan.get(time, "f_F1_1W").flatten()  # type: ignore
        if not np.isnan(f_F1_1W_val).any():
            f_l1_pos = p_WF1_val + base_foot_vertices[0]
            f_l1_val = f_F1_1W_val * FORCE_SCALE
            force_l1.set_positions(posA=f_l1_pos, posB=(f_l1_pos + f_l1_val))
            force_l1.set_visible(True)
        else:
            force_l1.set_visible(False)

        f_F1_2W_val = plan.get(time, "f_F1_2W").flatten()  # type: ignore
        if not np.isnan(f_F1_2W_val).any():
            f_l2_pos = p_WF1_val + base_foot_vertices[1]
            f_l2_val = f_F1_2W_val * FORCE_SCALE
            force_l2.set_positions(posA=f_l2_pos, posB=(f_l2_pos + f_l2_val))
            force_l2.set_visible(True)
        else:
            force_l2.set_visible(False)

        # Forces for right foot
        f_F2_1W_val = plan.get(time, "f_F2_1W").flatten()  # type: ignore
        if f_F2_1W_val is not None and not np.isnan(f_F2_1W_val).any():
            f_r1_pos = p_WF2_val + base_foot_vertices[0]
            f_r1_val = f_F2_1W_val * FORCE_SCALE
            force_r1.set_positions(posA=f_r1_pos, posB=(f_r1_pos + f_r1_val))
            force_r1.set_visible(True)
        else:
            force_r1.set_visible(False)

        f_F2_2W_val = plan.get(time, "f_F2_2W").flatten()  # type: ignore
        if f_F2_2W_val is not None and not np.isnan(f_F2_2W_val).any():
            f_r2_pos = p_WF2_val + base_foot_vertices[1]
            f_r2_val = f_F2_2W_val * FORCE_SCALE
            force_r2.set_positions(posA=f_r2_pos, posB=(f_r2_pos + f_r2_val))
            force_r2.set_visible(True)
        else:
            force_r2.set_visible(False)

    # Create and display animation
    n_steps = plan.num_steps + 1
    ani = FuncAnimation(fig, animate, frames=n_steps, interval=plan.dt * 1000)  # type: ignore
    if output_file is not None:
        ani.save(f"{output_file}.mp4", writer="ffmpeg")

    return ani
