from typing import Literal, Optional

import numpy as np
import numpy.typing as npt
from pydrake.solvers import MathematicalProgramResult

from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    CollisionGeometry,
    PolytopeContactLocation,
)
from planning_through_contact.geometry.hyperplane import Hyperplane
from planning_through_contact.geometry.planar.non_collision import NonCollisionVariables
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.planar.planar_pushing_trajectory import (
    PlanarPushingTrajectory,
)
from planning_through_contact.geometry.planar.trajectory_builder import (
    OldPlanarPushingTrajectory,
)
from planning_through_contact.planning.planar.planar_pushing_planner import (
    PlanarPushingPlanner,
)


def _get_p_and_R(traj: PlanarPushingTrajectory, idx: int):
    knot_points = traj.path_knot_points[idx]
    if isinstance(knot_points, NonCollisionVariables):
        R_WB = knot_points.R_WB
        p_WB = knot_points.p_WB
    else:
        R_WB = knot_points.R_WBs[idx]
        p_WB = knot_points.p_WBs[idx]

    return p_WB, R_WB


def _assert_traj_slider_pose(
    traj: PlanarPushingTrajectory,
    target_pose: PlanarPose,
    start_or_end: Literal["start", "end"],
    atol: float = 1e-3,
):
    if start_or_end == "start":
        idx = 0
    else:  # end
        idx = -1
    p_WB, R_WB = _get_p_and_R(traj, idx)

    assert np.allclose(R_WB, target_pose.two_d_rot_matrix(), atol=atol)
    assert np.allclose(p_WB, target_pose.pos(), atol=atol)


def _assert_traj_finger_pos(
    traj: PlanarPushingTrajectory,
    target_pose_slider: PlanarPose,
    target_pose_finger: PlanarPose,
    start_or_end: Literal["start", "end"],
    body_frame: bool = False,
):
    if start_or_end == "start":
        t = traj.start_time
    else:  # end
        t = traj.end_time

    if body_frame:
        finger_pos = traj.get_value(t, "p_BP")
    else:  # world frame
        finger_pos = traj.get_value(t, "p_WP")

    assert np.allclose(finger_pos, target_pose_finger.pos())


def assert_initial_and_final_poses(
    traj: PlanarPushingTrajectory,
    initial_slider_pose: Optional[PlanarPose],
    initial_finger_pose: Optional[PlanarPose],
    target_slider_pose: Optional[PlanarPose],
    target_finger_pose: Optional[PlanarPose],
    body_frame: bool = False,
) -> None:
    if initial_slider_pose:
        _assert_traj_slider_pose(traj, initial_slider_pose, "start")

    if initial_finger_pose and initial_slider_pose:
        _assert_traj_finger_pos(
            traj, initial_slider_pose, initial_finger_pose, "start", body_frame
        )

    if target_slider_pose:
        _assert_traj_slider_pose(traj, target_slider_pose, "end")

    if target_finger_pose and target_slider_pose:
        _assert_traj_finger_pos(
            traj, target_slider_pose, target_finger_pose, "end", body_frame
        )


def _assert_traj_slider_pose_legacy(
    traj: OldPlanarPushingTrajectory,
    target_pose: PlanarPose,
    start_or_end: Literal["start", "end"],
    atol: float = 1e-3,
):
    if start_or_end == "start":
        assert np.allclose(traj.R_WB[0], target_pose.two_d_rot_matrix(), atol=atol)
        assert np.allclose(traj.p_WB[:, 0:1], target_pose.pos(), atol=atol)
    else:  # end
        assert np.allclose(traj.R_WB[-1], target_pose.two_d_rot_matrix(), atol=atol)
        assert np.allclose(traj.p_WB[:, -1:], target_pose.pos(), atol=atol)


def _assert_traj_finger_pos_legacy(
    traj: OldPlanarPushingTrajectory,
    target_pose_slider: PlanarPose,
    target_pose_finger: PlanarPose,
    start_or_end: Literal["start", "end"],
    atol: float = 1e-3,
    relative_to_W: bool = False,
):
    if relative_to_W:
        p_c_W_target = target_pose_finger.pos()
    else:
        p_c_W_target = (
            target_pose_slider.pos()
            + target_pose_slider.two_d_rot_matrix().dot(target_pose_finger.pos())
        )
    if start_or_end == "start":
        assert np.allclose(traj.p_WP[:, 0:1], p_c_W_target, atol=atol)
    else:  # end
        assert np.allclose(traj.p_WP[:, -1:], p_c_W_target, atol=atol)


def assert_initial_and_final_poses_LEGACY(
    traj: OldPlanarPushingTrajectory,
    initial_slider_pose: Optional[PlanarPose],
    initial_finger_pose: Optional[PlanarPose],
    target_slider_pose: Optional[PlanarPose],
    target_finger_pose: Optional[PlanarPose],
    relative_to_W: bool = False,
) -> None:
    if initial_slider_pose:
        _assert_traj_slider_pose_legacy(traj, initial_slider_pose, "start")

    if initial_finger_pose and initial_slider_pose:
        _assert_traj_finger_pos_legacy(
            traj, initial_slider_pose, initial_finger_pose, "start", relative_to_W
        )

    if target_slider_pose:
        _assert_traj_slider_pose_legacy(traj, target_slider_pose, "end")

    if target_finger_pose and target_slider_pose:
        _assert_traj_finger_pos_legacy(
            traj, target_slider_pose, target_finger_pose, "end", relative_to_W
        )


def assert_object_is_avoided(
    object_geometry: CollisionGeometry,
    finger_traj: npt.NDArray[np.float64],  # (num_dims, traj_length)
    min_distance: float = 0.05,
    start_idx: int = 1,
    end_idx: int = -1,
) -> None:
    """
    Checks that all finger positions in a trajectory is outside at least one face of the object.

    @param start_idx: Where to start checking. By default, the first point is skipped (as it will often be in contact)
    @param end_idx: Where to stop checking. By default, the last point is skipped (as it will often be in contact)
    """
    outside_faces = [
        # each point must be outside one face
        any([face.dist_to(p_BF) >= min_distance for face in object_geometry.faces])
        for p_BF in finger_traj.T[start_idx:end_idx]
    ]
    assert all(outside_faces)
