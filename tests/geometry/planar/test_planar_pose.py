import numpy as np
import pytest
from pydrake.math import RigidTransform, RollPitchYaw

from geometry.planar.planar_pose import PlanarPose


@pytest.fixture
def object_height() -> float:
    return 0.2


@pytest.fixture
def planar_pose() -> PlanarPose:
    x = 0.2
    y = 0.5
    theta = 0.3
    return PlanarPose(x, y, theta)


def test_from_pose(planar_pose: PlanarPose, object_height: float) -> None:
    pose = RigidTransform(
        RollPitchYaw(np.array([np.pi, 0.0, planar_pose.theta])),  # type: ignore
        np.array([planar_pose.x, planar_pose.y, object_height]),
    )
    target_pose = PlanarPose.from_pose(pose)

    assert planar_pose.x == target_pose.x
    assert planar_pose.y == target_pose.y
    assert np.isclose(planar_pose.theta, target_pose.theta)


def test_to_pose(planar_pose: PlanarPose, object_height: float) -> None:
    pose = planar_pose.to_pose(object_height)

    target_rotation = (
        RollPitchYaw(np.array([np.pi, 0.0, planar_pose.theta]))
        .ToRotationMatrix()
        .matrix()
    )
    target_translation = (np.array([planar_pose.x, planar_pose.y, object_height]),)

    assert np.allclose(pose.rotation().matrix(), target_rotation)
    assert np.allclose(pose.translation(), target_translation)


def test_from_gen_coords(planar_pose: PlanarPose, object_height: float) -> None:
    transform = planar_pose.to_pose(object_height)
    quat = transform.rotation().ToQuaternion().wxyz()
    trans = transform.translation()
    gen_coords = np.concatenate((quat, trans))

    new_planar_pose = PlanarPose.from_generalized_coords(gen_coords)

    assert np.isclose(planar_pose.x, new_planar_pose.x)
    assert np.isclose(planar_pose.y, new_planar_pose.y)
    assert np.isclose(planar_pose.theta, new_planar_pose.theta)


def test_to_gen_coords(planar_pose: PlanarPose, object_height: float) -> None:
    transform = planar_pose.to_pose(object_height)
    quat = transform.rotation().ToQuaternion().wxyz()
    trans = transform.translation()
    gen_coords = np.concatenate((quat, trans))

    new_gen_coords = planar_pose.to_generalized_coords(object_height)

    assert np.allclose(gen_coords, new_gen_coords)


def test_vector(planar_pose: PlanarPose) -> None:
    target = np.array([planar_pose.x, planar_pose.y, planar_pose.theta])
    assert np.allclose(planar_pose.vector(), target)


def test_pos(planar_pose: PlanarPose) -> None:
    target = np.array([planar_pose.x, planar_pose.y]).reshape((2, 1))
    assert np.allclose(planar_pose.pos(), target)


def test_full_vector(planar_pose: PlanarPose) -> None:
    target = np.array(
        [
            planar_pose.x,
            planar_pose.y,
            np.cos(planar_pose.theta),
            np.sin(planar_pose.theta),
        ]
    )
    assert np.allclose(planar_pose.full_vector(), target)


def test_two_d_rot_matrix(planar_pose: PlanarPose) -> None:
    R_target = np.array([[0.9553364891, -0.2955202067], [0.2955202067, 0.9553364891]])
    assert np.allclose(planar_pose.two_d_rot_matrix(), R_target)
