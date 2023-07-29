from typing import Literal, Optional

import numpy as np
import numpy.typing as npt

from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    CollisionGeometry,
    PolytopeContactLocation,
)
from planning_through_contact.geometry.hyperplane import Hyperplane
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.planar.trajectory_builder import PlanarTrajectory


def _assert_traj_slider_pose(
    traj: PlanarTrajectory,
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


def _assert_traj_finger_pos(
    traj: PlanarTrajectory,
    target_pose_slider: PlanarPose,
    target_pose_finger: PlanarPose,
    start_or_end: Literal["start", "end"],
    atol: float = 1e-3,
):
    p_c_W_target = target_pose_slider.pos() + target_pose_slider.two_d_rot_matrix().dot(
        target_pose_finger.pos()
    )
    if start_or_end == "start":
        assert np.allclose(traj.p_c_W[:, 0:1], p_c_W_target, atol=atol)
    else:  # end
        assert np.allclose(traj.p_c_W[:, -1:], p_c_W_target, atol=atol)


def assert_initial_and_final_poses(
    traj: PlanarTrajectory,
    initial_slider_pose: Optional[PlanarPose],
    initial_finger_pose: Optional[PlanarPose],
    target_slider_pose: Optional[PlanarPose],
    target_finger_pose: Optional[PlanarPose],
) -> None:
    if initial_slider_pose:
        _assert_traj_slider_pose(traj, initial_slider_pose, "start")

    if initial_finger_pose and initial_slider_pose:
        _assert_traj_finger_pos(traj, initial_slider_pose, initial_finger_pose, "start")

    if target_slider_pose:
        _assert_traj_slider_pose(traj, target_slider_pose, "end")

    if target_finger_pose and target_slider_pose:
        _assert_traj_finger_pos(traj, target_slider_pose, target_finger_pose, "end")


def assert_object_is_avoided(
    object_geometry: CollisionGeometry,
    traj: PlanarTrajectory,
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
        for p_BF in traj.p_c_B.T[start_idx:end_idx]
    ]
    assert all(outside_faces)
