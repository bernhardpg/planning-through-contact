import numpy as np
from pydrake.solvers import MathematicalProgramResult

from planning_through_contact.planning.in_plane.contact_scene_program import (
    ContactSceneProgram,
)
from planning_through_contact.planning.in_plane.in_plane_trajectory import (
    InPlaneTrajectory,
)
from planning_through_contact.visualize.colors import COLORS
from planning_through_contact.visualize.visualizer_2d import (
    VisualizationPoint2d,
    VisualizationPolygon2d,
    Visualizer2d,
)


def visualize_in_plane_manipulation_plan(
    result: MathematicalProgramResult, problem: ContactSceneProgram
) -> None:
    bodies = problem.contact_scene_def.rigid_bodies

    traj = InPlaneTrajectory.create_from_result(result, problem)

    CONTACT_COLOR = COLORS["dodgerblue4"]
    GRAVITY_COLOR = COLORS["blueviolet"]
    BOX_COLOR = COLORS["aquamarine4"]
    TABLE_COLOR = COLORS["bisque3"]
    FINGER_COLOR = COLORS["firebrick3"]

    body_colors = [TABLE_COLOR, BOX_COLOR, FINGER_COLOR]

    viz_com_points = [
        VisualizationPoint2d(com, GRAVITY_COLOR)  # type: ignore
        for com in traj.body_positions.values()
    ]
    viz_polygons = [
        VisualizationPolygon2d.from_trajs(
            np.hstack(traj.body_positions[body]).T,
            np.hstack(traj.get_flat_body_rotations(body)).T,
            body.geometry,
            color,
        )
        for body, color in zip(bodies, body_colors)
    ]

    viz = Visualizer2d(PLOT_SCALE=1000, FORCE_SCALE=0.25, POINT_RADIUS=0.005)

    frames_per_sec = 1

    viz.visualize(
        [] + viz_com_points,
        [],
        viz_polygons,
        frames_per_sec,
        None,
    )
