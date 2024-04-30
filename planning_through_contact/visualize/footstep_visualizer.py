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
) -> None:
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
    p_WFl = ax.scatter(0, 0, color="b", zorder=3, label="Left foot")
    p_WFr = ax.scatter(0, 0, color="g", zorder=3, label="Right foot")

    # Misc settings
    plt.close()
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.3), ncol=2)

    def animate(n_steps: int) -> None:
        # Robot position and orientation
        if not np.isnan(plan.knot_points.p_WB[n_steps]).any():
            p_WB.set_offsets(plan.knot_points.p_WB[n_steps])
            robot_body.set_center(plan.knot_points.p_WB[n_steps])
            robot_body.angle = plan.knot_points.theta_WB[n_steps] * 180 / np.pi
            p_WB.set_visible(True)
            robot_body.set_visible(True)
        else:
            p_WB.set_visible(False)
            robot_body.set_visible(False)

        # Left foot
        if not np.isnan(plan.knot_points.p_WFl[n_steps]).any():
            foot_left.set_xy(base_foot_vertices + plan.knot_points.p_WFl[n_steps])
            p_WFl.set_offsets(plan.knot_points.p_WFl[n_steps])
            foot_left.set_visible(True)
            p_WFl.set_visible(True)
        else:
            foot_left.set_visible(False)
            p_WFl.set_visible(False)

        # Right foot
        if not np.isnan(plan.knot_points.p_WFr[n_steps]).any():
            foot_right.set_xy(base_foot_vertices + plan.knot_points.p_WFr[n_steps])
            p_WFr.set_offsets(plan.knot_points.p_WFr[n_steps])
            foot_right.set_visible(True)
            p_WFr.set_visible(True)
        else:
            foot_right.set_visible(False)
            p_WFr.set_visible(False)

        # Forces for left foot
        if not np.isnan(plan.knot_points.f_Fl_1W[n_steps]).any():
            f_l1_pos = plan.knot_points.p_WFl[n_steps] + base_foot_vertices[0]
            f_l1_val = plan.knot_points.f_Fl_1W[n_steps] * FORCE_SCALE
            force_l1.set_positions(posA=f_l1_pos, posB=(f_l1_pos + f_l1_val))
            force_l1.set_visible(True)
        else:
            force_l1.set_visible(False)

        if not np.isnan(plan.knot_points.f_Fl_2W[n_steps]).any():
            f_l2_pos = plan.knot_points.p_WFl[n_steps] + base_foot_vertices[1]
            f_l2_val = plan.knot_points.f_Fl_2W[n_steps] * FORCE_SCALE
            force_l2.set_positions(posA=f_l2_pos, posB=(f_l2_pos + f_l2_val))
            force_l2.set_visible(True)
        else:
            force_l2.set_visible(False)

        # Forces for right foot
        if not np.isnan(plan.knot_points.f_Fr_1W[n_steps]).any():  # type: ignore
            f_r1_pos = plan.knot_points.p_WFr[n_steps] + base_foot_vertices[0]
            f_r1_val = plan.knot_points.f_Fr_1W[n_steps] * FORCE_SCALE  # type: ignore
            force_r1.set_positions(posA=f_r1_pos, posB=(f_r1_pos + f_r1_val))
            force_r1.set_visible(True)
        else:
            force_r1.set_visible(False)

        if not np.isnan(plan.knot_points.f_Fr_2W[n_steps]).any():  # type: ignore
            f_r2_pos = plan.knot_points.p_WFr[n_steps] + base_foot_vertices[1]
            f_r2_val = plan.knot_points.f_Fr_2W[n_steps] * FORCE_SCALE  # type: ignore
            force_r2.set_positions(posA=f_r2_pos, posB=(f_r2_pos + f_r2_val))
            force_r2.set_visible(True)
        else:
            force_r2.set_visible(False)

    # Create and display animation
    n_steps = plan.num_steps
    ani = FuncAnimation(fig, animate, frames=n_steps, interval=1e3)  # type: ignore
    if output_file is not None:
        ani.save(f"{output_file}.mp4", writer="ffmpeg")
