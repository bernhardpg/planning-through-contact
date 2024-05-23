from typing import List

import numpy as np
import numpy.typing as npt
from pydrake.solvers import MathematicalProgramResult

from planning_through_contact.geometry.in_plane.contact_scene import FrictionConeDetails
from planning_through_contact.planning.in_plane.contact_scene_program import (
    ContactSceneProgram,
)
from planning_through_contact.planning.in_plane.in_plane_trajectory import (
    InPlaneTrajectory,
)
from planning_through_contact.visualize.colors import COLORS
from planning_through_contact.visualize.visualizer_2d import (
    VisualizationForce2d,
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

    viz_com_points = [VisualizationPoint2d(com, GRAVITY_COLOR) for com in traj.body_positions.values()]  # type: ignore
    viz_polygons = [
        VisualizationPolygon2d.from_trajs(
            np.hstack(traj.body_positions[body]).T,
            np.hstack(traj.get_flat_body_rotations(body)).T,
            body.geometry,
            color,
        )
        for body, color in zip(bodies, body_colors)
    ]

    viz_contact_forces = [
        VisualizationForce2d(np.hstack(pos).T, CONTACT_COLOR, np.hstack(force).T)
        for pos, force in zip(
            traj.contact_positions.values(), traj.contact_forces.values()
        )
    ]

    viz_gravity_forces = [
        VisualizationForce2d(
            np.hstack(traj.body_positions[body]).T,
            CONTACT_COLOR,
            np.hstack(traj.gravity_forces[body]).T,
        )
        for body in problem.contact_scene_def.unactuated_bodies
    ]

    def _get_fc_positions(
        cone_ctrl_points: List[FrictionConeDetails],
    ) -> List[npt.NDArray[np.float64]]:
        return [fc.p_WFc_W for fc in cone_ctrl_points]  # type: ignore

    def _get_fc_rotations(
        cone_ctrl_points: List[FrictionConeDetails],
    ) -> List[npt.NDArray[np.float64]]:
        return [fc.R_WFc for fc in cone_ctrl_points]  # type: ignore

    # viz_friction_cones = [
    #     VisualizationCone2d.from_ctrl_points(
    #         np.hstack(_get_fc_positions(fc)),
    #         _get_fc_rotations(fc),
    #         fc[0].normal_vec_local,
    #         np.arctan(fc[0].friction_coeff),
    #     )
    #     for fc in traj.friction_cones.values()
    # ]

    # viz_friction_cones_mirrored = [
    #     VisualizationCone2d.from_ctrl_points(
    #         evaluate_np_expressions_array(pos, motion_plan.result),
    #         [
    #             motion_plan.result.GetSolution(R_ctrl_point)
    #             for R_ctrl_point in orientation
    #         ],
    #         -normal_vec,
    #         friction_cone_angle,
    #     )
    #     for normal_vec, pos, orientation in zip(
    #         fc_normals, fc_positions, fc_orientations
    #     )
    # ]

    viz = Visualizer2d(PLOT_SCALE=1000, FORCE_SCALE=0.25, POINT_RADIUS=0.005)

    frames_per_sec = 1

    viz.visualize(
        [] + viz_com_points,
        viz_contact_forces + viz_gravity_forces,
        viz_polygons,
        frames_per_sec,
        None,
    )
