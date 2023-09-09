import numpy as np

from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    CollisionGeometry,
)
from planning_through_contact.geometry.planar.trajectory_builder import (
    OldPlanarPushingTrajectory,
)
from planning_through_contact.visualize.colors import COLORS
from planning_through_contact.visualize.visualizer_2d import (
    VisualizationForce2d,
    VisualizationPoint2d,
    VisualizationPolygon2d,
    Visualizer2d,
)


def visualize_planar_pushing_trajectory(
    traj: OldPlanarPushingTrajectory,
    object_geometry: CollisionGeometry,
    pusher_radius: float,
    visualize_object_vel: bool = False,
    visualize_robot_base: bool = False,
) -> None:
    CONTACT_COLOR = COLORS["dodgerblue4"]
    BOX_COLOR = COLORS["aquamarine4"]
    FINGER_COLOR = COLORS["firebrick3"]
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

    contact_point_viz = VisualizationPoint2d(traj.p_c_W.T, FINGER_COLOR)
    contact_point_viz.change_radius(pusher_radius)

    contact_force_viz = VisualizationForce2d(traj.p_c_W.T, CONTACT_COLOR, traj.f_c_W.T)
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
