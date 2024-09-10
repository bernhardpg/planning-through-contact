from pathlib import Path
from typing import Any, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from pydrake.common.value import Value
from pydrake.geometry import Box as DrakeBox
from pydrake.geometry import Convex
from pydrake.geometry import Cylinder as DrakeCylinder
from pydrake.geometry import (
    FrameId,
    FramePoseVector,
    GeometryFrame,
    GeometryInstance,
    MakePhongIllustrationProperties,
    SceneGraph,
)
from pydrake.geometry import Sphere as DrakeSphere
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.systems.all import Context, DiagramBuilder, LeafSystem
from pydrake.systems.analysis import Simulator
from pydrake.systems.planar_scenegraph_visualizer import (
    ConnectPlanarSceneGraphVisualizer,
    PlanarSceneGraphVisualizer,
)

from planning_through_contact.geometry.collision_geometry.box_2d import Box2d
from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    CollisionGeometry,
)
from planning_through_contact.geometry.collision_geometry.t_pusher_2d import TPusher2d
from planning_through_contact.geometry.collision_geometry.vertex_defined_geometry import (
    VertexDefinedGeometry,
)
from planning_through_contact.geometry.planar.face_contact import FaceContactVariables
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.planar.planar_pushing_trajectory import (
    FaceContactTrajSegment,
    NonCollisionTrajSegment,
    PlanarPushingTrajectory,
    SimplePlanarPushingTrajectory,
)
from planning_through_contact.geometry.planar.trajectory_builder import (
    OldPlanarPushingTrajectory,
)
from planning_through_contact.geometry.utilities import two_d_rotation_matrix_from_angle
from planning_through_contact.planning.planar.planar_plan_config import (
    PlanarPlanConfig,
    PlanarPushingStartAndGoal,
)
from planning_through_contact.visualize.colors import (
    AQUAMARINE4,
    BLACK,
    CADMIUMORANGE,
    COLORS,
    CRIMSON,
    DARKORCHID2,
    EMERALDGREEN,
    RGB,
)
from planning_through_contact.visualize.visualizer_2d import (
    VisualizationForce2d,
    VisualizationPoint2d,
    VisualizationPolygon2d,
    Visualizer2d,
)


def compare_trajs_vertically(
    trajs: List[PlanarPushingTrajectory],
    filename: Optional[str] = None,
    plot_lims: Optional[Tuple[float, float, float, float]] = None,
    plot_knot_points: bool = True,
    legends: Optional[List[str]] = None,
) -> None:
    # We need to add the first vertex again to close the polytope
    traj = trajs[0]
    vertices = np.hstack(
        traj.config.slider_geometry.vertices + [traj.config.slider_geometry.vertices[0]]
    )

    get_vertices_W = lambda p_WB, R_WB: p_WB + R_WB.dot(vertices)

    # Colors for each subplot
    colors = ["red", "blue", "green", "purple", "orange"]

    GOAL_COLOR = EMERALDGREEN.diffuse()
    GOAL_TRANSPARENCY = 1.0

    START_COLOR = CRIMSON.diffuse()
    START_TRANSPARENCY = 1.0

    def _get_seg_groups(
        traj,
    ) -> List[
        List[
            Tuple[
                FaceContactTrajSegment | NonCollisionTrajSegment, FaceContactVariables
            ]
        ]
    ]:
        segment_groups = []
        idx = 0
        while idx < len(traj.path_knot_points):
            group = []
            no_face_contact = True
            while no_face_contact and idx < len(traj.path_knot_points):
                curr_segment = traj.traj_segments[idx]
                curr_knot_points = traj.path_knot_points[idx]
                group.append((curr_segment, curr_knot_points))
                idx += 1

                if isinstance(curr_knot_points, FaceContactVariables):
                    no_face_contact = False

            segment_groups.append(group)

        # Remove the last group as we don't need to plot this
        # (slider is standing still, only pusher is moving)
        if len(segment_groups) > 1:
            segment_groups = segment_groups[:-1]

        return segment_groups

    seg_groups = [_get_seg_groups(traj) for traj in trajs]
    max_length = max(len(group) for group in seg_groups)

    fig_height = 5
    fig, axs = plt.subplots(
        len(trajs), max_length, figsize=(fig_height * max_length, fig_height)
    )

    if plot_lims is not None:
        x_min, x_max, y_min, y_max = plot_lims
    else:
        x_min, x_max, y_min, y_max = traj.get_pos_limits(buffer=0.5)

    for ax_row in axs:
        for ax in ax_row:
            ax.axis("equal")  # Ensures the x and y axis are scaled equally

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            # Hide the axes, including the spines, ticks, labels, and title
            ax.set_axis_off()

    def _plot_segment_groups(row_idx, groups, color, color_dark):
        for segment_idx, segment_group in enumerate(groups):
            ax = axs[row_idx, segment_idx]

            num_frames_in_group = sum(
                [knot_points.num_knot_points for _, knot_points in segment_group]
            )
            frame_count = 0

            start_transparency = 0.5
            end_transparency = 0.9
            get_transp_for_frame = (
                lambda idx, num_points: (end_transparency - start_transparency)
                * idx
                / num_points
                + start_transparency
            )

            for element_idx, (traj_segment, knot_points) in enumerate(segment_group):
                ts = np.linspace(
                    traj_segment.start_time,
                    traj_segment.end_time,
                    knot_points.num_knot_points,
                )

                for idx in range(knot_points.num_knot_points):
                    R_WB = traj_segment.get_R_WB(ts[idx])[:2, :2]  # 2x2 matrix
                    p_WB = traj_segment.get_p_WB(ts[idx])
                    p_WP = traj_segment.get_p_WP(ts[idx])

                    # We only plot the current frame if it will change next frame
                    # (this is to avoid plotting multiple frames on top of each other)
                    if idx + 1 < knot_points.num_knot_points:
                        next_R_WB = traj_segment.get_R_WB(ts[idx + 1])[
                            :2, :2
                        ]  # 2x2 matrix
                        next_p_WB = traj_segment.get_p_WB(ts[idx + 1])
                        next_p_WP = traj_segment.get_p_WP(ts[idx + 1])
                    else:
                        next_R_WB = R_WB
                        next_p_WB = p_WB
                        next_p_WP = p_WP

                    vertices_W = get_vertices_W(p_WB, R_WB)

                    transparency = get_transp_for_frame(
                        frame_count, num_frames_in_group
                    )
                    line_transparency = transparency

                    # Plot polytope
                    if (
                        np.any(next_R_WB != R_WB)
                        or np.any(next_p_WB != p_WB)
                        or element_idx == len(segment_group) - 1
                    ):
                        ax.plot(
                            vertices_W[0, :],
                            vertices_W[1, :],
                            color=color,
                            alpha=line_transparency
                            * 0.7,  # we plot the slider a bit more transparent to make forces visible
                            linewidth=1,
                        )

                    # # Plot pusher
                    # if np.any(next_p_WP != p_WP) or element_idx == len(segment_group) - 1:
                    #     ax.add_patch(make_circle(p_WP, fill_transparency))

                    # Plot forces
                    FORCE_SCALE = 2.0
                    FORCE_VIS_TRESH = 1e-4
                    # only N-1 inputs
                    if (idx < knot_points.num_knot_points - 1) and (
                        isinstance(knot_points, FaceContactVariables)
                    ):
                        f_W = traj_segment.get_f_W(ts[idx]).flatten()
                        if np.linalg.norm(f_W) < FORCE_VIS_TRESH:
                            continue

                        p_Wc = traj_segment.get_p_Wc(ts[idx]).flatten()
                        ax.arrow(
                            p_Wc[0],
                            p_Wc[1],
                            f_W[0] * FORCE_SCALE,
                            f_W[1] * FORCE_SCALE,
                            color=color_dark,
                            fill=True,
                            zorder=99999,
                            alpha=line_transparency,
                            joinstyle="round",
                            linewidth=0.0,
                            width=0.006,
                        )

                    frame_count += 1

    for idx, (color, group) in enumerate(zip(colors, seg_groups)):
        _plot_segment_groups(idx, group, color, color)

    if not plot_knot_points:
        raise NotImplementedError(
            "Support for making figure of interpolated trajectory is not yet supported"
        )

    # Create a list of patches to use as legend handles
    if legends is not None:
        custom_patches = [
            mpatches.Patch(color=color, label=label)
            for label, color in zip(legends, colors)
        ]
        # Creating the custom legend
        plt.legend(handles=custom_patches)

    fig.tight_layout()
    if filename is not None:
        fig.savefig(filename + f".pdf")  # type: ignore
    else:
        plt.show()


def compare_trajs(
    traj_a: PlanarPushingTrajectory,
    traj_b: PlanarPushingTrajectory,
    filename: Optional[str] = None,
    plot_lims: Optional[Tuple[float, float, float, float]] = None,
    plot_knot_points: bool = True,
    traj_a_legend: str = "traj_a",
    traj_b_legend: str = "traj_b",
) -> None:
    # We need to add the first vertex again to close the polytope
    vertices = np.hstack(
        traj_a.config.slider_geometry.vertices
        + [traj_a.config.slider_geometry.vertices[0]]
    )

    get_vertices_W = lambda p_WB, R_WB: p_WB + R_WB.dot(vertices)

    TRAJ_A_COLOR = COLORS["aquamarine4"].diffuse()
    TRAJ_A_COLOR_DARK = COLORS["darkgreen"].diffuse()
    TRAJ_B_COLOR = COLORS["firebrick3"].diffuse()
    TRAJ_B_COLOR_DARK = COLORS["firebrick4"].diffuse()

    GOAL_COLOR = EMERALDGREEN.diffuse()
    GOAL_TRANSPARENCY = 1.0

    START_COLOR = CRIMSON.diffuse()
    START_TRANSPARENCY = 1.0

    traj_a_segment_groups = []
    idx = 0
    while idx < len(traj_a.path_knot_points):
        group = []
        no_face_contact = True
        while no_face_contact and idx < len(traj_a.path_knot_points):
            curr_segment = traj_a.traj_segments[idx]
            curr_knot_points = traj_a.path_knot_points[idx]
            group.append((curr_segment, curr_knot_points))
            idx += 1

            if isinstance(curr_knot_points, FaceContactVariables):
                no_face_contact = False

        traj_a_segment_groups.append(group)

    traj_b_segment_groups = []
    idx = 0
    while idx < len(traj_b.path_knot_points):
        group = []
        no_face_contact = True
        while no_face_contact and idx < len(traj_b.path_knot_points):
            curr_segment = traj_b.traj_segments[idx]
            curr_knot_points = traj_b.path_knot_points[idx]
            group.append((curr_segment, curr_knot_points))
            idx += 1

            if isinstance(curr_knot_points, FaceContactVariables):
                no_face_contact = False

        traj_b_segment_groups.append(group)

    # Remove the last group as we don't need to plot this
    # (slider is standing still, only pusher is moving)
    if len(traj_a_segment_groups) > 1:
        traj_a_segment_groups = traj_a_segment_groups[:-1]
    if len(traj_b_segment_groups) > 1:
        traj_b_segment_groups = traj_b_segment_groups[:-1]

    fig_height = 5
    fig, axs = plt.subplots(
        1,
        len(traj_a_segment_groups),
        figsize=(fig_height * len(traj_a_segment_groups), fig_height),
    )

    if plot_lims is not None:
        x_min, x_max, y_min, y_max = plot_lims
    else:
        x_min, x_max, y_min, y_max = traj_a.get_pos_limits(buffer=0.5)

    def _plot_segment_groups(segment_groups, color, color_dark):
        for segment_idx, segment_group in enumerate(segment_groups):
            if len(segment_groups) == 1:  # only one subplot
                ax = axs
            else:
                ax = axs[segment_idx]

            ax.axis("equal")  # Ensures the x and y axis are scaled equally

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            # Hide the axes, including the spines, ticks, labels, and title
            ax.set_axis_off()

            num_frames_in_group = sum(
                [knot_points.num_knot_points for _, knot_points in segment_group]
            )
            frame_count = 0

            start_transparency = 0.5
            end_transparency = 0.9
            get_transp_for_frame = (
                lambda idx, num_points: (end_transparency - start_transparency)
                * idx
                / num_points
                + start_transparency
            )

            for element_idx, (traj_segment, knot_points) in enumerate(segment_group):
                ts = np.linspace(
                    traj_segment.start_time,
                    traj_segment.end_time,
                    knot_points.num_knot_points,
                )

                for idx in range(knot_points.num_knot_points):
                    R_WB = traj_segment.get_R_WB(ts[idx])[:2, :2]  # 2x2 matrix
                    p_WB = traj_segment.get_p_WB(ts[idx])
                    p_WP = traj_segment.get_p_WP(ts[idx])

                    # We only plot the current frame if it will change next frame
                    # (this is to avoid plotting multiple frames on top of each other)
                    if idx + 1 < knot_points.num_knot_points:
                        next_R_WB = traj_segment.get_R_WB(ts[idx + 1])[
                            :2, :2
                        ]  # 2x2 matrix
                        next_p_WB = traj_segment.get_p_WB(ts[idx + 1])
                        next_p_WP = traj_segment.get_p_WP(ts[idx + 1])
                    else:
                        next_R_WB = R_WB
                        next_p_WB = p_WB
                        next_p_WP = p_WP

                    vertices_W = get_vertices_W(p_WB, R_WB)

                    transparency = get_transp_for_frame(
                        frame_count, num_frames_in_group
                    )
                    line_transparency = transparency

                    # Plot polytope
                    if (
                        np.any(next_R_WB != R_WB)
                        or np.any(next_p_WB != p_WB)
                        or element_idx == len(segment_group) - 1
                    ):
                        ax.plot(
                            vertices_W[0, :],
                            vertices_W[1, :],
                            color=color,
                            alpha=line_transparency
                            * 0.7,  # we plot the slider a bit more transparent to make forces visible
                            linewidth=1,
                        )

                    # # Plot pusher
                    # if np.any(next_p_WP != p_WP) or element_idx == len(segment_group) - 1:
                    #     ax.add_patch(make_circle(p_WP, fill_transparency))

                    # Plot forces
                    FORCE_SCALE = 2.0
                    FORCE_VIS_TRESH = 1e-4
                    # only N-1 inputs
                    if (idx < knot_points.num_knot_points - 1) and (
                        isinstance(knot_points, FaceContactVariables)
                    ):
                        f_W = traj_segment.get_f_W(ts[idx]).flatten()
                        if np.linalg.norm(f_W) < FORCE_VIS_TRESH:
                            continue

                        p_Wc = traj_segment.get_p_Wc(ts[idx]).flatten()
                        ax.arrow(
                            p_Wc[0],
                            p_Wc[1],
                            f_W[0] * FORCE_SCALE,
                            f_W[1] * FORCE_SCALE,
                            color=color_dark,
                            fill=True,
                            zorder=99999,
                            alpha=line_transparency,
                            joinstyle="round",
                            linewidth=0.0,
                            width=0.006,
                        )

                    frame_count += 1

    _plot_segment_groups(traj_a_segment_groups, TRAJ_A_COLOR, TRAJ_A_COLOR_DARK)
    _plot_segment_groups(traj_b_segment_groups, TRAJ_B_COLOR, TRAJ_B_COLOR_DARK)

    if not plot_knot_points:
        raise NotImplementedError(
            "Support for making figure of interpolated trajectory is not yet supported"
        )

        # # Plot start pos
        # slider_initial_pose = traj_a.config.start_and_goal.slider_initial_pose  # type: ignore
        # p_WB = slider_initial_pose.pos()
        # R_WB = slider_initial_pose.two_d_rot_matrix()
        # goal_vertices_W = get_vertices_W(p_WB, R_WB)
        # ax.plot(
        #     goal_vertices_W[0, :],
        #     goal_vertices_W[1, :],
        #     color=START_COLOR,
        #     alpha=START_TRANSPARENCY,
        #     linewidth=1,
        #     linestyle="--",
        # )
        # if traj_a.config.start_and_goal.pusher_initial_pose is not None:
        #     p_WP = traj_a.config.start_and_goal.pusher_initial_pose.pos()  # type: ignore
        #     circle = plt.Circle(
        #         p_WP.flatten(),
        #         traj_a.config.pusher_radius,  # type: ignore
        #         edgecolor=START_COLOR,
        #         facecolor="none",
        #         linewidth=1,
        #         alpha=START_TRANSPARENCY,
        #         linestyle="--",
        #     )
        #     ax.add_patch(circle)
        #
        # # Plot target pos
        # slider_target_pose = traj_a.config.start_and_goal.slider_target_pose  # type: ignore
        # p_WB = slider_target_pose.pos()
        # R_WB = slider_target_pose.two_d_rot_matrix()
        # goal_vertices_W = get_vertices_W(p_WB, R_WB)
        # ax.plot(
        #     goal_vertices_W[0, :],
        #     goal_vertices_W[1, :],
        #     color=GOAL_COLOR,
        #     alpha=GOAL_TRANSPARENCY,
        #     linewidth=1,
        #     linestyle="--",
        # )
        # if traj_a.config.start_and_goal.pusher_initial_pose is not None:
        #     p_WP = traj_a.config.start_and_goal.pusher_target_pose.pos()  # type: ignore
        #     circle = plt.Circle(
        #         p_WP.flatten(),
        #         traj_a.config.pusher_radius,  # type: ignore
        #         edgecolor=GOAL_COLOR,
        #         facecolor="none",
        #         linewidth=1,
        #         alpha=GOAL_TRANSPARENCY,
        #         linestyle="--",
        #     )
        #     ax.add_patch(circle)

    # Create a list of patches to use as legend handles
    custom_patches = [
        mpatches.Patch(color=color, label=label)
        for label, color in zip(
            [traj_a_legend, traj_b_legend], [TRAJ_A_COLOR, TRAJ_B_COLOR]
        )
    ]
    # Creating the custom legend
    plt.legend(handles=custom_patches)

    fig.tight_layout()
    if filename is not None:
        fig.savefig(filename + f".pdf")  # type: ignore
        plt.close()


def plot_forces(
    traj: PlanarPushingTrajectory,
    filename: str,
) -> None:
    face_knot_points = [
        knot_points
        for knot_points in traj.path_knot_points
        if isinstance(knot_points, FaceContactVariables)
    ]

    normal_forces = np.concatenate(
        [knot_point.normal_forces for knot_point in face_knot_points]
    )
    friction_forces = np.concatenate(
        [knot_point.friction_forces for knot_point in face_knot_points]
    )

    fig, axs = plt.subplots(2)
    # First plot
    axs[0].plot(normal_forces)
    axs[0].set_title("Normal forces")

    # Second plot
    axs[1].plot(friction_forces)
    axs[1].set_title("Friction forces")

    # Automatically adjust subplot params so that the subplot(s) fits in to the figure area
    plt.tight_layout()

    if filename is not None:
        fig.savefig(filename + ".pdf")  # type: ignore
        plt.close()
    else:
        plt.show()


def plot_simple_traj(
    traj: SimplePlanarPushingTrajectory,
    filename: Optional[str] = None,
    plot_lims: Optional[Tuple[float, float, float, float]] = None,
    start_end_legend: bool = False,
    slider_color: Optional[Any] = None,
    keyframe_times: Optional[List[float]] = None,
    times_for_keyframes: Optional[list[int]] = None,
    num_keyframes: int = 5,
    num_times_per_keyframe: int = 5,
    label: Optional[str] = None,
) -> None:
    # Ensure a type 1 font is used
    plt.rcParams["font.family"] = "Times"
    plt.rcParams["ps.useafm"] = True
    plt.rcParams["pdf.use14corefonts"] = True
    plt.rcParams["text.usetex"] = False
    # NOTE(bernhardpg): This function is a mess!
    # We need to add the first vertex again to close the polytope
    vertices = np.hstack(
        traj.config.slider_geometry.vertices + [traj.config.slider_geometry.vertices[0]]
    )

    get_vertices_W = lambda p_WB, R_WB: p_WB + R_WB.dot(vertices)

    if slider_color is None:
        slider_color = COLORS["aquamarine4"].diffuse()

    PUSHER_COLOR = COLORS["firebrick3"].diffuse()
    # PUSHER_COLOR = CADMIUMORANGE.diffuse()

    LINE_COLOR = BLACK.diffuse()

    GOAL_COLOR = EMERALDGREEN.diffuse()
    GOAL_TRANSPARENCY = 1.0

    START_COLOR = CRIMSON.diffuse()
    START_TRANSPARENCY = 1.0

    if keyframe_times is None:
        keyframe_times = np.linspace(traj.start_time, traj.end_time, num_keyframes + 1)  # type: ignore

    assert keyframe_times is not None

    fig_height = 4
    fig, axs = plt.subplots(
        1, num_keyframes, figsize=(fig_height * num_keyframes, fig_height)
    )

    if plot_lims is not None:
        x_min, x_max, y_min, y_max = plot_lims
    else:
        x_min, x_max, y_min, y_max = traj.get_pos_limits(buffer=0.12)

    for keyframe_idx, (keyframe_t_curr, keyframe_t_next) in enumerate(
        zip(keyframe_times[:-1], keyframe_times[1:])
    ):
        if times_for_keyframes is not None:
            num_times_per_keyframe = times_for_keyframes[keyframe_idx]

        times = np.linspace(keyframe_t_curr, keyframe_t_next, num_times_per_keyframe)  # type: ignore

        ax = axs[keyframe_idx]

        ax.axis("equal")  # Ensures the x and y axis are scaled equally

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        # Hide the axes, including the spines, ticks, labels, and title
        ax.set_axis_off()

        frame_count = 0

        make_circle = lambda p_WP, fill_transparency: plt.Circle(
            p_WP.flatten(),
            traj.config.pusher_radius,  # type: ignore
            edgecolor=LINE_COLOR,
            facecolor=PUSHER_COLOR,
            linewidth=1,
            alpha=fill_transparency,
            zorder=99,
        )

        start_transparency = 0.3
        end_transparency = 1.0
        get_transp_for_frame = (
            lambda idx, num_points: (end_transparency - start_transparency)
            * idx
            / num_points
            + start_transparency
        )

        for t in times:
            p_WP = traj.get_value(t, "p_WP")
            num_frames_in_keyframe = len(times)
            transparency = get_transp_for_frame(frame_count, num_frames_in_keyframe)
            if transparency > 1:
                breakpoint()
            ax.add_patch(make_circle(p_WP, transparency))

            # get the constant slider pose
            R_WB = traj.get_value(t, "R_WB")[:2, :2]  # 2x2 matrix
            p_WB = traj.get_value(t, "p_WB")

            # Plot polytope
            vertices_W = get_vertices_W(p_WB, R_WB)
            ax.plot(
                vertices_W[0, :],
                vertices_W[1, :],
                color=LINE_COLOR,
                alpha=transparency,
                linewidth=1,
            )
            ax.fill(
                vertices_W[0, :],
                vertices_W[1, :],
                alpha=transparency,
                color=slider_color,
            )

            frame_count += 1

        # Plot start pos
        slider_initial_pose = traj.config.start_and_goal.slider_initial_pose  # type: ignore
        p_WB = slider_initial_pose.pos()
        R_WB = slider_initial_pose.two_d_rot_matrix()
        goal_vertices_W = get_vertices_W(p_WB, R_WB)
        start_goal_width = 1.5
        ax.plot(
            goal_vertices_W[0, :],
            goal_vertices_W[1, :],
            color=START_COLOR,
            alpha=START_TRANSPARENCY,
            linewidth=start_goal_width,
            linestyle="--",
        )
        assert traj.config.start_and_goal is not None
        if traj.config.start_and_goal.pusher_initial_pose is not None:
            p_WP = traj.config.start_and_goal.pusher_initial_pose.pos()  # type: ignore
            circle = plt.Circle(
                p_WP.flatten(),
                traj.config.pusher_radius,  # type: ignore
                edgecolor=START_COLOR,
                facecolor="none",
                linewidth=start_goal_width,
                alpha=START_TRANSPARENCY,
                linestyle="--",
            )
            ax.add_patch(circle)

        # Plot target pos
        slider_target_pose = traj.config.start_and_goal.slider_target_pose  # type: ignore
        p_WB = slider_target_pose.pos()
        R_WB = slider_target_pose.two_d_rot_matrix()
        goal_vertices_W = get_vertices_W(p_WB, R_WB)
        ax.plot(
            goal_vertices_W[0, :],
            goal_vertices_W[1, :],
            color=GOAL_COLOR,
            alpha=GOAL_TRANSPARENCY,
            linewidth=start_goal_width,
            linestyle="--",
        )
        if traj.config.start_and_goal.pusher_initial_pose is not None:
            p_WP = traj.config.start_and_goal.pusher_target_pose.pos()  # type: ignore
            circle = plt.Circle(
                p_WP.flatten(),
                traj.config.pusher_radius,  # type: ignore
                edgecolor=GOAL_COLOR,
                facecolor="none",
                linewidth=start_goal_width,
                alpha=GOAL_TRANSPARENCY,
                linestyle="--",
            )
            ax.add_patch(circle)

        if start_end_legend:
            # Create a list of patches to use as legend handles
            custom_patches = [
                mpatches.Patch(color=color, label=label)
                for label, color in zip(["Start", "Goal"], [START_COLOR, GOAL_COLOR])
            ]
            # Creating the custom legend
            plt.legend(
                handles=custom_patches, handlelength=2.5, fontsize=22, loc="upper right"
            )

        if label:
            # Create a list of patches to use as legend handles
            patch = mpatches.Patch(label=label, color="None")
            # Creating the custom legend
            plt.legend(handles=[patch], handlelength=2.5, fontsize=22, loc="upper left")

    fig.tight_layout()
    if filename:
        fig.savefig(filename + f"_trajectory.pdf", format="pdf")  # type: ignore
        plt.close()
    else:
        plt.show()


def make_traj_figure(
    traj: PlanarPushingTrajectory,
    filename: Optional[str] = None,
    plot_lims: Optional[Tuple[float, float, float, float]] = None,
    plot_knot_points: bool = True,
    show_workspace: bool = False,
    start_end_legend: bool = False,
    plot_forces: bool = True,
    slider_color: Optional[Any] = None,
    split_on_mode_type: bool = False,
    num_contact_frames: int = 5,
    num_non_collision_frames: int = 5,
) -> None:
    # Ensure a type 1 font is used
    plt.rcParams["font.family"] = "Times"
    plt.rcParams["ps.useafm"] = True
    plt.rcParams["pdf.use14corefonts"] = True
    plt.rcParams["text.usetex"] = False
    # NOTE(bernhardpg): This function is a mess!
    # We need to add the first vertex again to close the polytope
    vertices = np.hstack(
        traj.config.slider_geometry.vertices + [traj.config.slider_geometry.vertices[0]]
    )

    get_vertices_W = lambda p_WB, R_WB: p_WB + R_WB.dot(vertices)

    if slider_color is None:
        slider_color = COLORS["aquamarine4"].diffuse()

    PUSHER_COLOR = COLORS["firebrick3"].diffuse()
    # PUSHER_COLOR = CADMIUMORANGE.diffuse()

    LINE_COLOR = BLACK.diffuse()

    GOAL_COLOR = EMERALDGREEN.diffuse()
    GOAL_TRANSPARENCY = 1.0

    START_COLOR = CRIMSON.diffuse()
    START_TRANSPARENCY = 1.0

    segment_groups = []
    if split_on_mode_type:
        prev_mode = traj.path_knot_points[0]
        idx = 0
        while idx < len(traj.path_knot_points):
            group = []
            new_mode_type = False
            while not new_mode_type and idx < len(traj.path_knot_points):
                curr_segment = traj.traj_segments[idx]
                curr_knot_points = traj.path_knot_points[idx]

                if not isinstance(curr_knot_points, type(prev_mode)):
                    new_mode_type = True
                    prev_mode = curr_knot_points
                else:
                    group.append((curr_segment, curr_knot_points))
                    idx += 1

            segment_groups.append(group)
    else:
        idx = 0
        while idx < len(traj.path_knot_points):
            group = []
            no_face_contact = True
            while no_face_contact and idx < len(traj.path_knot_points):
                curr_segment = traj.traj_segments[idx]
                curr_knot_points = traj.path_knot_points[idx]
                group.append((curr_segment, curr_knot_points))
                idx += 1

                if isinstance(curr_knot_points, FaceContactVariables):
                    no_face_contact = False

            segment_groups.append(group)

    fig_height = 4
    fig, axs = plt.subplots(
        1, len(segment_groups), figsize=(fig_height * len(segment_groups), fig_height)
    )

    if plot_lims is not None:
        x_min, x_max, y_min, y_max = plot_lims
    else:
        x_min, x_max, y_min, y_max = traj.get_pos_limits(buffer=0.12)

    for segment_idx, segment_group in enumerate(segment_groups):
        if len(segment_groups) == 1:  # only one subplot
            ax = axs
        else:
            ax = axs[segment_idx]

        ax.axis("equal")  # Ensures the x and y axis are scaled equally

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        # Hide the axes, including the spines, ticks, labels, and title
        ax.set_axis_off()

        num_frames_in_group = sum(
            [knot_points.num_knot_points for _, knot_points in segment_group]
        )
        frame_count = 0

        make_circle = lambda p_WP, fill_transparency: plt.Circle(
            p_WP.flatten(),
            traj.config.pusher_radius,  # type: ignore
            edgecolor=LINE_COLOR,
            facecolor=PUSHER_COLOR,
            linewidth=1,
            alpha=fill_transparency,
        )

        start_transparency = 0.3
        end_transparency = 1.0
        get_transp_for_frame = (
            lambda idx, num_points: (end_transparency - start_transparency)
            * idx
            / num_points
            + start_transparency
        )

        if show_workspace:
            # Plot workspace
            ws_rect = plt.Rectangle(
                (0.4, -0.25),
                0.35,
                0.5,
                linewidth=1,
                edgecolor="grey",
                facecolor="none",
                linestyle="--",
            )
            ax.add_patch(ws_rect)

        # count how many frames we have plotted in this group
        frame_count = 0
        for element_idx, (traj_segment, knot_points) in enumerate(segment_group):
            # we don't plot the first mode (source)
            num_segments_to_plot = len(
                [
                    1
                    for _, knot_points in segment_group
                    if knot_points.num_knot_points > 1
                ]
            )

            if not plot_knot_points:
                if knot_points.num_knot_points == 1 and element_idx == 0:
                    # do not plot first mode (source vertex) as it
                    # looks weird
                    continue

                if isinstance(traj_segment, NonCollisionTrajSegment):
                    num_frames_in_segment = num_non_collision_frames

                    if segment_idx < len(segment_groups) - 1:
                        num_frames_in_group = (
                            num_frames_in_segment * num_segments_to_plot
                        )
                    else:  # last group, we plot last mode
                        num_frames_in_group = (
                            num_frames_in_segment * (len(segment_group) - 1) + 1
                        )
                else:  # face contact
                    num_frames_in_segment = num_contact_frames
                    # Only one face contact
                    num_frames_in_group = num_frames_in_segment

                if (
                    segment_idx == len(segment_groups) - 1
                    and element_idx == len(segment_group) - 1
                ):
                    ts = [traj_segment.start_time]
                else:
                    ts = np.linspace(
                        traj_segment.start_time,
                        traj_segment.end_time,
                        num_frames_in_segment,
                    )
                if element_idx < len(segment_group) - 1:
                    ts = ts[
                        :-1
                    ]  # avoid plotting the last pusher of intermittent modes so we don't get double plots

                for idx, t in enumerate(ts):
                    p_WP = traj_segment.get_p_WP(t)
                    transparency = get_transp_for_frame(
                        frame_count, num_frames_in_group
                    )
                    if transparency > 1:
                        breakpoint()
                    ax.add_patch(make_circle(p_WP, transparency))

                    # get the constant slider pose
                    R_WB = traj_segment.get_R_WB(t)[:2, :2]  # 2x2 matrix
                    p_WB = traj_segment.get_p_WB(t)

                    # Plot polytope
                    vertices_W = get_vertices_W(p_WB, R_WB)
                    ax.plot(
                        vertices_W[0, :],
                        vertices_W[1, :],
                        color=LINE_COLOR,
                        alpha=transparency,
                        linewidth=1,
                    )
                    ax.fill(
                        vertices_W[0, :],
                        vertices_W[1, :],
                        alpha=transparency,
                        color=slider_color,
                    )

                    if plot_forces:
                        # Plot forces
                        FORCE_SCALE = 8.0
                        # only N-1 inputs
                        if (idx < knot_points.num_knot_points - 1) and (
                            isinstance(knot_points, FaceContactVariables)
                        ):
                            f_W = traj_segment.get_f_W(ts[idx]).flatten() * FORCE_SCALE
                            p_Wc = traj_segment.get_p_Wc(ts[idx]).flatten()
                            ax.arrow(
                                p_Wc[0],
                                p_Wc[1],
                                f_W[0],
                                f_W[1],
                                color=LINE_COLOR,
                                fill=True,
                                zorder=99999,
                                alpha=transparency,
                                joinstyle="round",
                                linewidth=0.0,
                                width=0.008,
                            )

                    frame_count += 1

            else:  # plot knot points values directly (which will shrink object if rotations have det R < 1 etc.)
                ts = np.linspace(
                    traj_segment.start_time,
                    traj_segment.end_time,
                    knot_points.num_knot_points,
                )

                for idx in range(knot_points.num_knot_points):
                    R_WB = traj_segment.get_R_WB(ts[idx])[:2, :2]  # 2x2 matrix
                    p_WB = traj_segment.get_p_WB(ts[idx])

                    p_WP = traj_segment.get_p_WP(ts[idx])

                    # This is to get the knot point value which can have determinant < 1
                    if len(traj_segment.R_WB.Rs) > 1:
                        R_WB = traj_segment.R_WB.Rs[idx]
                    else:
                        R_WB = traj_segment.R_WB.Rs[0]

                    # We only plot the current frame if it will change next frame
                    # (this is to avoid plotting multiple frames on top of each other)
                    if idx + 1 < knot_points.num_knot_points:
                        next_R_WB = traj_segment.get_R_WB(ts[idx + 1])[
                            :2, :2
                        ]  # 2x2 matrix
                        next_p_WB = traj_segment.get_p_WB(ts[idx + 1])
                        next_p_WP = traj_segment.get_p_WP(ts[idx + 1])
                    else:
                        next_R_WB = R_WB
                        next_p_WB = p_WB
                        next_p_WP = p_WP

                    vertices_W = get_vertices_W(p_WB, R_WB)

                    transparency = get_transp_for_frame(
                        frame_count, num_frames_in_group
                    )
                    line_transparency = transparency
                    fill_transparency = transparency

                    # Plot polytope
                    if (
                        np.any(next_R_WB != R_WB)
                        or np.any(next_p_WB != p_WB)
                        or element_idx == len(segment_group) - 1
                    ):
                        ax.plot(
                            vertices_W[0, :],
                            vertices_W[1, :],
                            color=LINE_COLOR,
                            alpha=line_transparency,
                            linewidth=1,
                        )
                        ax.fill(
                            vertices_W[0, :],
                            vertices_W[1, :],
                            alpha=fill_transparency,
                            color=slider_color,
                        )

                    # Plot pusher
                    if (
                        np.any(next_p_WP != p_WP)
                        or element_idx == len(segment_group) - 1
                    ):
                        ax.add_patch(make_circle(p_WP, fill_transparency))

                    if plot_forces:
                        # Plot forces
                        FORCE_SCALE = 0.5
                        # only N-1 inputs
                        if (idx < knot_points.num_knot_points - 1) and (
                            isinstance(knot_points, FaceContactVariables)
                        ):
                            f_W = traj_segment.get_f_W(ts[idx]).flatten() * FORCE_SCALE

                            TOL = 1e-3
                            if np.linalg.norm(f_W) <= TOL:
                                # don't plot zero forces
                                continue

                            p_Wc = traj_segment.get_p_Wc(ts[idx]).flatten()
                            ax.arrow(
                                p_Wc[0],
                                p_Wc[1],
                                f_W[0],
                                f_W[1],
                                color=LINE_COLOR,
                                fill=True,
                                zorder=99999,
                                alpha=line_transparency,
                                joinstyle="round",
                                linewidth=0.0,
                                width=0.008,
                            )

                        frame_count += 1

        # Plot start pos
        slider_initial_pose = traj.config.start_and_goal.slider_initial_pose  # type: ignore
        p_WB = slider_initial_pose.pos()
        R_WB = slider_initial_pose.two_d_rot_matrix()
        goal_vertices_W = get_vertices_W(p_WB, R_WB)
        start_goal_width = 1.5
        ax.plot(
            goal_vertices_W[0, :],
            goal_vertices_W[1, :],
            color=START_COLOR,
            alpha=START_TRANSPARENCY,
            linewidth=start_goal_width,
            linestyle="--",
        )
        if traj.config.start_and_goal.pusher_initial_pose is not None:
            p_WP = traj.config.start_and_goal.pusher_initial_pose.pos()  # type: ignore
            circle = plt.Circle(
                p_WP.flatten(),
                traj.config.pusher_radius,  # type: ignore
                edgecolor=START_COLOR,
                facecolor="none",
                linewidth=start_goal_width,
                alpha=START_TRANSPARENCY,
                linestyle="--",
            )
            ax.add_patch(circle)

        # Plot target pos
        slider_target_pose = traj.config.start_and_goal.slider_target_pose  # type: ignore
        p_WB = slider_target_pose.pos()
        R_WB = slider_target_pose.two_d_rot_matrix()
        goal_vertices_W = get_vertices_W(p_WB, R_WB)
        ax.plot(
            goal_vertices_W[0, :],
            goal_vertices_W[1, :],
            color=GOAL_COLOR,
            alpha=GOAL_TRANSPARENCY,
            linewidth=start_goal_width,
            linestyle="--",
        )
        if traj.config.start_and_goal.pusher_initial_pose is not None:
            p_WP = traj.config.start_and_goal.pusher_target_pose.pos()  # type: ignore
            circle = plt.Circle(
                p_WP.flatten(),
                traj.config.pusher_radius,  # type: ignore
                edgecolor=GOAL_COLOR,
                facecolor="none",
                linewidth=start_goal_width,
                alpha=GOAL_TRANSPARENCY,
                linestyle="--",
            )
            ax.add_patch(circle)

        if start_end_legend:
            # Create a list of patches to use as legend handles
            custom_patches = [
                mpatches.Patch(color=color, label=label)
                for label, color in zip(["Start", "Goal"], [START_COLOR, GOAL_COLOR])
            ]
            # Creating the custom legend
            axs[0].legend(
                handles=custom_patches,
                handlelength=2.5,
                fontsize=22,
                loc="upper left",
            )

    fig.tight_layout()
    if filename:
        fig.savefig(filename + f"_trajectory.pdf", format="pdf")  # type: ignore
        plt.close()
    else:
        plt.show()


def _create_polygon_mesh(
    vertices: npt.NDArray[np.float64], filename: str = "polygon.obj"
) -> None:
    # Ensure vertices are in the correct format (N x 3 for 3D mesh)
    if vertices.shape[1] != 3:
        raise ValueError("Vertices should have 3 columns for x, y, z coordinates.")

    # Write vertices to an OBJ file
    with open(filename, "w") as file:
        file.write("# OBJ file\n")
        for vertex in vertices:
            file.write("v {} {} {}\n".format(vertex[0], vertex[1], vertex[2]))

        # Assuming the polygon is a closed loop and vertices are ordered
        file.write("f")
        for i in range(len(vertices)):
            file.write(" {}".format(i + 1))
        file.write("\n")


def _load_2d_vertices_as_mesh(
    vertices: List[npt.NDArray[np.float64]],
) -> Convex:
    def _make_3d(v: npt.NDArray[np.float64], height: float) -> npt.NDArray[np.float64]:
        return np.array([v[0, 0], v[1, 0], height]).reshape((3, 1))

    HEIGHT = 0.3
    vertices_zero_height = [_make_3d(v, height=0) for v in vertices]
    vertices_fixed_height = [_make_3d(v, height=HEIGHT) for v in vertices]

    all_vertices = np.hstack(vertices_zero_height + vertices_fixed_height).T  # (N, 3)

    # Drake requires us to load the mesh from a file, so we make a temporary file which
    # we then delete
    temp_file = Path("temp/slider_geometry.obj")
    temp_file.parent.mkdir(exist_ok=True, parents=True)
    _create_polygon_mesh(all_vertices, str(temp_file))
    mesh = Convex(str(temp_file))

    # NOTE: The file can apparently not be deleted until the animation is completed,
    # so for now we just leave it.
    # temp_file.unlink()  # delete the file
    return mesh


def _add_slider_geometries(
    source_id,
    slider_geometry: CollisionGeometry,
    scene_graph: SceneGraph,
    slider_frame_id: FrameId,
    alpha: float = 1.0,
    color: RGB = AQUAMARINE4,
    show_com: bool = False,
) -> None:
    DEFAULT_HEIGHT = 0.3

    if isinstance(slider_geometry, Box2d):
        box_geometry_id = scene_graph.RegisterGeometry(
            source_id,
            slider_frame_id,
            GeometryInstance(
                RigidTransform.Identity(),
                DrakeBox(slider_geometry.width, slider_geometry.height, DEFAULT_HEIGHT),
                "slider",
            ),
        )
        scene_graph.AssignRole(
            source_id,
            box_geometry_id,
            MakePhongIllustrationProperties(color.diffuse(alpha)),
        )
    elif isinstance(slider_geometry, TPusher2d):
        boxes, transforms = slider_geometry.get_as_boxes(DEFAULT_HEIGHT / 2)
        box_geometry_ids = [
            scene_graph.RegisterGeometry(
                source_id,
                slider_frame_id,
                GeometryInstance(
                    transform,
                    DrakeBox(box.width, box.height, DEFAULT_HEIGHT),
                    f"box_{idx}",
                ),
            )
            for idx, (box, transform) in enumerate(zip(boxes, transforms))
        ]
        for box_geometry_id in box_geometry_ids:
            scene_graph.AssignRole(
                source_id,
                box_geometry_id,
                MakePhongIllustrationProperties(color.diffuse(alpha)),
            )
    elif isinstance(slider_geometry, VertexDefinedGeometry):
        mesh = _load_2d_vertices_as_mesh(slider_geometry.vertices)
        geometry_id = scene_graph.RegisterGeometry(
            source_id,
            slider_frame_id,
            GeometryInstance(RigidTransform.Identity(), mesh, "slider"),
        )
        scene_graph.AssignRole(
            source_id,
            geometry_id,
            MakePhongIllustrationProperties(color.diffuse(alpha)),
        )
    else:
        raise NotImplementedError(
            f"Cannot add geometry {slider_geometry.__class__.__name__} to builder."
        )

    if show_com:
        com_id = scene_graph.RegisterGeometry(
            source_id,
            slider_frame_id,
            GeometryInstance(
                RigidTransform(
                    RotationMatrix.Identity(), np.array([0, 0, 0])  # type: ignore
                ),
                DrakeSphere(0.005),
                "pusher",
            ),
        )
        com_color = BLACK.diffuse(alpha)
        scene_graph.AssignRole(
            source_id, com_id, MakePhongIllustrationProperties(com_color)
        )


def _add_pusher_geometry(
    source_id,
    pusher_radius: float,
    scene_graph: SceneGraph,
    pusher_frame_id: FrameId,
    alpha: float = 1.0,
) -> None:
    CYLINDER_HEIGHT = 0.3
    pusher_geometry_id = scene_graph.RegisterGeometry(
        source_id,
        pusher_frame_id,
        GeometryInstance(
            RigidTransform(
                RotationMatrix.Identity(), np.array([0, 0, CYLINDER_HEIGHT / 2])  # type: ignore
            ),
            DrakeCylinder(pusher_radius, CYLINDER_HEIGHT),
            "pusher",
        ),
    )
    pusher_COLOR = COLORS["firebrick3"]
    scene_graph.AssignRole(
        source_id,
        pusher_geometry_id,
        MakePhongIllustrationProperties(pusher_COLOR.diffuse(alpha)),
    )


class PlanarPushingStartGoalGeometry(LeafSystem):
    """
    Class used to visualize start and goal positions for a planar pushing task
    """

    def __init__(
        self,
        slider_initial_pose: PlanarPose,
        slider_target_pose: PlanarPose,
        pusher_initial_pose: PlanarPose,
        pusher_target_pose: PlanarPose,
        slider_geometry: CollisionGeometry,
        pusher_radius: float,
        scene_graph: SceneGraph,
    ) -> None:
        super().__init__()

        self.slider_initial_pose = slider_initial_pose
        self.slider_target_pose = slider_target_pose
        self.pusher_initial_pose = self._rotate_to_world(
            pusher_initial_pose, slider_initial_pose
        )
        self.pusher_target_pose = self._rotate_to_world(
            pusher_target_pose, slider_target_pose
        )

        self.DeclareAbstractOutputPort(
            "geometry_pose",
            alloc=lambda: Value(FramePoseVector()),
            calc=self.calc_output,  # type: ignore
        )

        self.source_id = scene_graph.RegisterSource()

        TRANSPARENCY = 0.3

        self.slider_frame_id = scene_graph.RegisterFrame(
            self.source_id, GeometryFrame("slider_start")
        )
        _add_slider_geometries(
            self.source_id,
            slider_geometry,
            scene_graph,
            self.slider_frame_id,
            alpha=TRANSPARENCY,
            show_com=False,
        )
        self.pusher_frame_id = scene_graph.RegisterFrame(
            self.source_id,
            GeometryFrame("pusher_start"),
        )
        _add_pusher_geometry(
            self.source_id,
            pusher_radius,
            scene_graph,
            self.pusher_frame_id,
            alpha=TRANSPARENCY,
        )

    @staticmethod
    def _rotate_to_world(
        pusher_pose: PlanarPose, slider_pose: PlanarPose
    ) -> PlanarPose:
        p_WP = pusher_pose.pos()
        R_WB = slider_pose.two_d_rot_matrix()
        p_WB = slider_pose.pos()

        # We need to compute the pusher pos in the frame of the slider
        p_BP = R_WB.T @ (p_WP - p_WB)
        pusher_pose_world = PlanarPose(p_BP[0, 0], p_BP[1, 0], 0)

        return pusher_pose_world

    @classmethod
    def add_to_builder(
        cls,
        builder: DiagramBuilder,
        slider_initial_pose: PlanarPose,
        slider_target_pose: PlanarPose,
        pusher_initial_pose: PlanarPose,
        pusher_target_pose: PlanarPose,
        slider_geometry: CollisionGeometry,
        pusher_radius: float,
        scene_graph: SceneGraph,
        name: str = "start_goal_geometry",
    ) -> "PlanarPushingTrajectory":
        traj_geometry = builder.AddNamedSystem(
            name,
            cls(
                slider_initial_pose,
                slider_target_pose,
                pusher_initial_pose,
                pusher_target_pose,
                slider_geometry,
                pusher_radius,
                scene_graph,
            ),
        )
        builder.Connect(
            traj_geometry.get_output_port(),
            scene_graph.get_source_pose_port(traj_geometry.source_id),
        )
        return traj_geometry

    def _set_outputs(
        self,
        slider_frame_id: FrameId,
        pusher_frame_id: FrameId,
        output: FramePoseVector,
        p_WB: npt.NDArray[np.float64],
        p_WP: npt.NDArray[np.float64],
        R_WB: npt.NDArray[np.float64],
    ):
        p_x = p_WB[0, 0]  # type: ignore
        p_y = p_WB[1, 0]  # type: ignore

        slider_pose = RigidTransform(
            RotationMatrix(R_WB), np.array([p_x, p_y, 0.0])  # type: ignore
        )
        output.get_mutable_value().set_value(id=slider_frame_id, value=slider_pose)  # type: ignore

        pusher_pose = RigidTransform(
            RotationMatrix.Identity(), np.concatenate((p_WP.flatten(), [0]))  # type: ignore
        )
        output.get_mutable_value().set_value(id=pusher_frame_id, value=pusher_pose)  # type: ignore

    def _get_pusher_in_world(self, slider_pose, pusher_pose) -> PlanarPose:
        p_WP = slider_pose.pos() + slider_pose.two_d_rot_matrix().dot(pusher_pose.pos())
        return PlanarPose(p_WP[0, 0], p_WP[1, 0], 0)

    def get_pos_limits(
        self, slider_geometry, buffer
    ) -> Tuple[float, float, float, float]:
        def get_lims(vecs) -> Tuple[float, float, float, float]:
            vec_xs = [vec[0, 0] for vec in vecs]
            vec_ys = [vec[1, 0] for vec in vecs]

            vec_x_max = max(vec_xs)
            vec_x_min = min(vec_xs)
            vec_y_max = max(vec_ys)
            vec_y_min = min(vec_ys)

            return vec_x_min, vec_x_max, vec_y_min, vec_y_max

        def add_buffer_to_lims(lims, buffer) -> Tuple[float, float, float, float]:
            return (
                lims[0] - buffer,
                lims[1] + buffer,
                lims[2] - buffer,
                lims[3] + buffer,
            )

        def get_lims_from_two_lims(lim_a, lim_b) -> Tuple[float, float, float, float]:
            return (
                min(lim_a[0], lim_b[0]),
                max(lim_a[1], lim_b[1]),
                min(lim_a[2], lim_b[2]),
                max(lim_a[3], lim_b[3]),
            )

        p_WB_lims = get_lims(
            [self.slider_initial_pose.pos(), self.slider_target_pose.pos()]
        )
        object_radius = slider_geometry.max_dist_from_com
        obj_lims = add_buffer_to_lims(p_WB_lims, object_radius)
        p_WP_lims = get_lims(
            [
                self._get_pusher_in_world(
                    self.slider_initial_pose, self.pusher_initial_pose
                ).pos(),
                self._get_pusher_in_world(
                    self.slider_target_pose, self.pusher_target_pose
                ).pos(),
            ]
        )

        lims = get_lims_from_two_lims(obj_lims, p_WP_lims)
        return add_buffer_to_lims(lims, buffer)

    def calc_output(self, context: Context, output: FramePoseVector) -> None:
        t = context.get_time()
        if t <= 1.0:
            slider_pose = self.slider_initial_pose
            pusher_pose = self.pusher_initial_pose
        else:
            slider_pose = self.slider_target_pose
            pusher_pose = self.pusher_target_pose

        self._set_outputs(
            self.slider_frame_id,
            self.pusher_frame_id,
            output,
            slider_pose.pos(),
            self._get_pusher_in_world(slider_pose, pusher_pose).pos(),
            slider_pose.rot_matrix(),
        )


class PlanarPushingTrajectoryGeometry(LeafSystem):
    def __init__(
        self,
        traj: PlanarPushingTrajectory,
        scene_graph: SceneGraph,
        visualize_knot_points: bool = False,
        visualize_goal: bool = True,
    ) -> None:
        super().__init__()

        self.traj = traj
        self.visualize_goal = visualize_goal
        self.visualize_knot_points = visualize_knot_points

        slider_geometry = self.traj.config.slider_geometry
        pusher_radius = self.traj.config.pusher_radius
        MIN_PUSHER_RADIUS_VIZ = 0.01  # we need some radius for the visualization
        if pusher_radius == 0:
            pusher_radius = MIN_PUSHER_RADIUS_VIZ

        self.DeclareAbstractOutputPort(
            "geometry_pose",
            alloc=lambda: Value(FramePoseVector()),
            calc=self.calc_output,  # type: ignore
        )

        self.source_id = scene_graph.RegisterSource()

        self.slider_frame_id = scene_graph.RegisterFrame(
            self.source_id, GeometryFrame("slider")
        )
        _add_slider_geometries(
            self.source_id,
            slider_geometry,
            scene_graph,
            self.slider_frame_id,
            show_com=True,
        )

        self.pusher_frame_id = scene_graph.RegisterFrame(
            self.source_id,
            GeometryFrame("pusher"),
        )
        _add_pusher_geometry(
            self.source_id, pusher_radius, scene_graph, self.pusher_frame_id
        )

        GOAL_TRANSPARENCY = 0.3
        if self.visualize_goal:
            self.slider_goal_frame_id = scene_graph.RegisterFrame(
                self.source_id, GeometryFrame("slider_goal")
            )
            _add_slider_geometries(
                self.source_id,
                slider_geometry,
                scene_graph,
                self.slider_goal_frame_id,
                alpha=GOAL_TRANSPARENCY,
                show_com=False,
            )
            self.pusher_goal_frame_id = scene_graph.RegisterFrame(
                self.source_id,
                GeometryFrame("pusher_goal"),
            )
            _add_pusher_geometry(
                self.source_id,
                pusher_radius,
                scene_graph,
                self.pusher_goal_frame_id,
                alpha=GOAL_TRANSPARENCY,
            )

        # TODO: Shows table
        # TABLE_COLOR = COLORS["bisque3"]
        # TABLE_HEIGHT = 0.1
        # table_geometry_id = scene_graph.RegisterAnchoredGeometry(
        #     self.source_id,
        #     GeometryInstance(
        #         RigidTransform(
        #             RotationMatrix.Identity(), np.array([0, 0, -TABLE_HEIGHT / 2])  # type: ignore
        #         ),
        #         DrakeBox(1.0, 1.0, TABLE_HEIGHT),
        #         "table",
        #     ),
        # )
        # scene_graph.AssignRole(
        #     self.source_id,
        #     table_geometry_id,
        #     MakePhongIllustrationProperties(TABLE_COLOR.diffuse()),
        # )

    @classmethod
    def add_to_builder(
        cls,
        builder: DiagramBuilder,
        traj: PlanarPushingTrajectory,
        scene_graph: SceneGraph,
        visualize_knot_points: bool = False,
        name: str = "traj_geometry ",
    ) -> "PlanarPushingTrajectory":
        traj_geometry = builder.AddNamedSystem(
            name,
            cls(
                traj,
                scene_graph,
                visualize_knot_points,
            ),
        )
        builder.Connect(
            traj_geometry.get_output_port(),
            scene_graph.get_source_pose_port(traj_geometry.source_id),
        )
        return traj_geometry

    def _set_outputs(
        self,
        slider_frame_id: FrameId,
        pusher_frame_id: FrameId,
        output: FramePoseVector,
        p_WB: npt.NDArray[np.float64],
        p_WP: npt.NDArray[np.float64],
        R_WB: npt.NDArray[np.float64],
    ):
        p_x = p_WB[0, 0]  # type: ignore
        p_y = p_WB[1, 0]  # type: ignore

        slider_pose = RigidTransform(
            RotationMatrix(R_WB), np.array([p_x, p_y, 0.0])  # type: ignore
        )
        output.get_mutable_value().set_value(id=slider_frame_id, value=slider_pose)  # type: ignore

        pusher_pose = RigidTransform(
            RotationMatrix.Identity(), np.concatenate((p_WP.flatten(), [0]))  # type: ignore
        )
        output.get_mutable_value().set_value(id=pusher_frame_id, value=pusher_pose)  # type: ignore

    def calc_output(self, context: Context, output: FramePoseVector) -> None:
        t = context.get_time()

        if self.visualize_knot_points:
            # TODO force
            R_WB = self.traj.get_knot_point_value(t, "R_WB")
            p_WB = self.traj.get_knot_point_value(t, "p_WB")
            p_WP = self.traj.get_knot_point_value(t, "p_WP")
        else:
            # TODO force
            R_WB = self.traj.get_value(t, "R_WB")
            p_WB = self.traj.get_value(t, "p_WB")
            p_WP = self.traj.get_value(t, "p_WP")

        self._set_outputs(
            self.slider_frame_id,
            self.pusher_frame_id,
            output,
            p_WB,  # type: ignore
            p_WP,  # type: ignore
            R_WB,  # type: ignore
        )

        if self.visualize_goal:
            target_slider_planar_pose = self.traj.target_slider_planar_pose
            target_pusher_planar_pose = self.traj.target_pusher_planar_pose
            if target_pusher_planar_pose is None:
                # If there is no pusher pose, we display the pos outside the frame
                target_pusher_pos = np.ones((2, 1)) * 99
            else:
                target_pusher_pos = target_pusher_planar_pose.pos()

            self._set_outputs(
                self.slider_goal_frame_id,
                self.pusher_goal_frame_id,
                output,
                target_slider_planar_pose.pos(),
                target_pusher_pos,
                target_slider_planar_pose.rot_matrix(),
            )


def visualize_planar_pushing_start_and_goal(
    slider_geometry: CollisionGeometry,
    pusher_radius: float,
    plan: PlanarPushingStartAndGoal,
    show: bool = False,
    save: bool = False,
    filename: Optional[str] = None,
):
    if save:
        assert filename is not None

    builder = DiagramBuilder()

    # Register geometry with SceneGraph
    scene_graph = builder.AddNamedSystem("scene_graph", SceneGraph())
    geometry = PlanarPushingStartGoalGeometry.add_to_builder(
        builder,
        plan.slider_initial_pose,
        plan.slider_target_pose,
        plan.pusher_initial_pose,
        plan.pusher_target_pose,
        slider_geometry,
        pusher_radius,
        scene_graph,
    )

    x_min, x_max, y_min, y_max = geometry.get_pos_limits(slider_geometry, buffer=0.1)

    def connect_planar_visualizer(
        builder: DiagramBuilder, scene_graph: SceneGraph
    ) -> PlanarSceneGraphVisualizer:
        T_VW = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        )
        visualizer = ConnectPlanarSceneGraphVisualizer(
            builder,
            scene_graph,
            T_VW=T_VW,
            xlim=np.array([x_min, x_max]),
            ylim=np.array([y_min, y_max]),
            show=show,
        )
        return visualizer

    visualizer = connect_planar_visualizer(builder, scene_graph)

    diagram = builder.Build()
    diagram.set_name("diagram")

    # Create the simulator, and simulate for 10 seconds.
    context = diagram.CreateDefaultContext()
    simulator = Simulator(diagram, context)

    visualizer.start_recording()  # type: ignore

    simulator.Initialize()
    simulator.AdvanceTo(2.0)

    visualizer.stop_recording()  # type: ignore
    ani = visualizer.get_recording_as_animation()  # type: ignore

    if save:
        # Playback the recording and save the output.
        ani.save(f"{filename}.mp4", fps=30)

    return ani


def visualize_planar_pushing_trajectory(
    traj: PlanarPushingTrajectory,
    show: bool = False,
    save: bool = False,
    filename: Optional[str] = None,
    visualize_knot_points: bool = False,
    lims: Optional[Tuple[float, float, float, float]] = None,
):
    if save:
        assert filename is not None

    builder = DiagramBuilder()

    # Register geometry with SceneGraph
    scene_graph = builder.AddNamedSystem("scene_graph", SceneGraph())
    traj_geometry = PlanarPushingTrajectoryGeometry.add_to_builder(
        builder,
        traj,
        scene_graph,
        visualize_knot_points,
    )

    if lims is None:
        x_min, x_max, y_min, y_max = traj.get_pos_limits(buffer=0.1)
    else:
        x_min, x_max, y_min, y_max = lims

    def connect_planar_visualizer(
        builder: DiagramBuilder, scene_graph: SceneGraph
    ) -> PlanarSceneGraphVisualizer:
        T_VW = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        )
        visualizer = ConnectPlanarSceneGraphVisualizer(
            builder,
            scene_graph,
            T_VW=T_VW,
            xlim=np.array([x_min, x_max]),
            ylim=np.array([y_min, y_max]),
            show=show,
        )
        return visualizer

    visualizer = connect_planar_visualizer(builder, scene_graph)

    diagram = builder.Build()
    diagram.set_name("diagram")

    # Create the simulator, and simulate for 10 seconds.
    context = diagram.CreateDefaultContext()
    simulator = Simulator(diagram, context)

    visualizer.start_recording()  # type: ignore

    simulator.Initialize()
    # simulator.set_target_realtime_rate(1.0)
    simulator.AdvanceTo(traj.end_time)

    visualizer.stop_recording()  # type: ignore
    ani = visualizer.get_recording_as_animation()  # type: ignore

    if save:
        # Playback the recording and save the output.
        ani.save(f"{filename}.mp4", fps=30)

    return ani


def visualize_planar_pushing_trajectory_legacy(
    traj: OldPlanarPushingTrajectory,
    object_geometry: CollisionGeometry,
    pusher_radius: float,
    visualize_object_vel: bool = False,
    visualize_robot_base: bool = False,
) -> None:
    CONTACT_COLOR = COLORS["dodgerblue4"]
    BOX_COLOR = COLORS["aquamarine4"]
    pusher_COLOR = COLORS["firebrick3"]
    TARGET_COLOR = COLORS["firebrick1"]

    flattened_rotation = np.vstack([R.flatten() for R in traj.R_WB])
    box_viz = VisualizationPolygon2d.from_trajs(
        traj.p_WB.T,
        flattened_rotation,
        object_geometry,
        BOX_COLOR,
    )

    # NOTE: I don't really need the entire trajectory here, but leave for now
    target_viz = VisualizationPolygon2d.from_trajs(
        traj.p_WB.T,
        flattened_rotation,
        object_geometry,
        TARGET_COLOR,
    )

    contact_point_viz = VisualizationPoint2d(traj.p_WP.T, pusher_COLOR)
    contact_point_viz.change_radius(pusher_radius)

    contact_force_viz = VisualizationForce2d(traj.p_WP.T, CONTACT_COLOR, traj.f_c_W.T)
    contact_forces_viz = [contact_force_viz]

    if visualize_object_vel:
        # TODO(bernhardpg): functionality that is useful for debugging
        v_WB = (traj.p_WB[:, 1:] - traj.p_WB[:, :-1]) / 0.1
        object_vel_viz = VisualizationForce2d(traj.p_WB.T, CONTACT_COLOR, v_WB.T)
        contact_forces_viz.append(
            object_vel_viz
        )  # visualize vel as a force (with an arrow)

    viz = Visualizer2d()
    FRAMES_PER_SEC = 1 / traj.dt
    viz.visualize(
        [contact_point_viz],
        contact_forces_viz,
        [box_viz],
        FRAMES_PER_SEC,
        target_viz,
        draw_origin=visualize_robot_base,
    )


def visualize_initial_conditions(
    initial_conditions: List[PlanarPushingStartAndGoal],
    config: PlanarPlanConfig,
    filename: Optional[str] = None,
    start_end_legend: bool = False,
    plot_orientation_arrow: bool = False,
) -> None:
    slider_target_pose = initial_conditions[0].slider_target_pose
    assert all(
        [conds.slider_target_pose == slider_target_pose for conds in initial_conditions]
    )
    pusher_initial_pose = initial_conditions[0].pusher_initial_pose
    assert all(
        [
            conds.pusher_initial_pose == pusher_initial_pose
            for conds in initial_conditions
        ]
    )
    pusher_target_pose = initial_conditions[0].pusher_target_pose
    assert all(
        [conds.pusher_target_pose == pusher_target_pose for conds in initial_conditions]
    )

    slider_initial_poses = [cond.slider_initial_pose for cond in initial_conditions]
    slider_geometry = config.slider_geometry

    # Ensure a type 1 font is used
    plt.rcParams["font.family"] = "Times"
    plt.rcParams["ps.useafm"] = True
    plt.rcParams["pdf.use14corefonts"] = True
    plt.rcParams["text.usetex"] = False
    # NOTE(bernhardpg): This function is a mess!
    # We need to add the first vertex again to close the polytope
    vertices = np.hstack(slider_geometry.vertices + [slider_geometry.vertices[0]])

    get_vertices_W = lambda p_WB, R_WB: p_WB + R_WB.dot(vertices)

    slider_color = COLORS["aquamarine4"].diffuse()

    GOAL_COLOR = EMERALDGREEN.diffuse()
    GOAL_TRANSPARENCY = 1.0

    START_COLOR = CRIMSON.diffuse()
    START_TRANSPARENCY = 1.0
    PUSHER_COLOR = COLORS["firebrick3"].diffuse()

    LINE_COLOR = BLACK.diffuse()

    fig_height = 4
    fig = plt.figure(figsize=(fig_height, fig_height))
    ax = fig.add_subplot(111)
    ax.axis("equal")  # Ensures the x and y axis are scaled equally

    # Hide the axes, including the spines, ticks, labels, and title
    ax.set_axis_off()

    make_circle = lambda p_WP, fill_transparency: plt.Circle(
        p_WP.flatten(),
        traj.config.pusher_radius,  # type: ignore
        edgecolor=LINE_COLOR,
        facecolor=PUSHER_COLOR,
        linewidth=1,
        alpha=fill_transparency,
        zorder=99,
    )

    transparency = 0.7

    def _plot_orientation_arrow(p_WB, R_WB, **kwargs):
        start_point = p_WB.flatten()
        u = np.array([0, 1]).reshape((2, 1)) / np.sqrt(2)
        u_rotated = R_WB @ u
        end_point = start_point + u_rotated.flatten() * 0.1

        # Plot the arrow
        ax.arrow(
            start_point[0],
            start_point[1],
            end_point[0] - start_point[0],
            end_point[1] - start_point[1],
            # head_width=2.0,
            # head_length=1.0,
            # fc="blue",
            # ec="black",
            **kwargs,
        )

    for pose in slider_initial_poses:
        # Plot polytope
        p_WB = pose.pos()
        R_WB = pose.two_d_rot_matrix()
        vertices_W = get_vertices_W(p_WB, R_WB)
        ax.plot(
            vertices_W[0, :],
            vertices_W[1, :],
            color=LINE_COLOR,
            alpha=transparency,
            linewidth=1,
        )
        ax.fill(
            vertices_W[0, :],
            vertices_W[1, :],
            alpha=transparency,
            color=slider_color,
        )

        if plot_orientation_arrow:
            _plot_orientation_arrow(p_WB, R_WB)

    start_goal_width = 1.5
    p_WP = pusher_initial_pose.pos()  # type: ignore
    circle = plt.Circle(
        p_WP.flatten(),
        config.pusher_radius,  # type: ignore
        edgecolor=GOAL_COLOR,
        facecolor="none",
        linewidth=start_goal_width,
        alpha=GOAL_TRANSPARENCY,
        linestyle="--",
    )
    ax.add_patch(circle)

    p_WB = slider_target_pose.pos()
    R_WB = slider_target_pose.two_d_rot_matrix()
    goal_vertices_W = get_vertices_W(p_WB, R_WB)
    ax.plot(
        goal_vertices_W[0, :],
        goal_vertices_W[1, :],
        color=GOAL_COLOR,
        alpha=GOAL_TRANSPARENCY,
        linewidth=start_goal_width,
        linestyle="--",
        zorder=99,
    )

    if plot_orientation_arrow:
        _plot_orientation_arrow(p_WB, R_WB, linestyle="--", color=GOAL_COLOR, zorder=99)

    if start_end_legend:
        # Create a list of patches to use as legend handles
        custom_patches = [
            mpatches.Patch(color=color, label=label)
            for label, color in zip(["Start", "Goal"], [START_COLOR, GOAL_COLOR])
        ]
        # Creating the custom legend
        plt.legend(
            handles=custom_patches, handlelength=2.5, fontsize=22, loc="upper right"
        )

    fig.tight_layout()
    if filename:
        fig.savefig(filename + ".pdf", format="pdf")  # type: ignore
        plt.close()
    else:
        plt.show()
