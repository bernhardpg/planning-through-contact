import numpy as np

from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    CollisionGeometry,
)
from planning_through_contact.geometry.planar.trajectory_builder import (
    PlanarPushingTrajectory,
)
from planning_through_contact.visualize.colors import COLORS
from planning_through_contact.visualize.visualizer_2d import (
    VisualizationForce2d,
    VisualizationPoint2d,
    VisualizationPolygon2d,
    Visualizer2d,
)


def visualize_planar_pushing_trajectory(
    traj: PlanarPushingTrajectory, object_geometry: CollisionGeometry
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
    contact_force_viz = VisualizationForce2d(traj.p_c_W.T, CONTACT_COLOR, traj.f_c_W.T)

    viz = Visualizer2d()
    FRAMES_PER_SEC = 1 / traj.dt
    viz.visualize(
        [contact_point_viz],
        [contact_force_viz],
        [box_viz],
        FRAMES_PER_SEC,
        target_viz,
    )
