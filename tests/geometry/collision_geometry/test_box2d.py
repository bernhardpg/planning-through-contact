import numpy as np
import pytest

from planning_through_contact.geometry.collision_geometry.box_2d import Box2d
from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    ContactLocation,
    PolytopeContactLocation,
)
from planning_through_contact.geometry.hyperplane import Hyperplane


@pytest.fixture
def box_geometry() -> Box2d:
    # Default sugar box dimensions
    width = 0.1703
    height = 0.0867
    box = Box2d(width, height)
    return box


def test_contact_locations(box_geometry: Box2d) -> None:
    locs = box_geometry.contact_locations
    assert len(locs) == 4

    # TODO This will likely change when I consider contacts with vertices
    assert np.all([loc.pos == ContactLocation.FACE for loc in locs])

    # Check that indices are 0, 1, 2, 3
    assert np.all([loc.idx == i for i, loc in enumerate(locs)])


def test_vertices(box_geometry: Box2d) -> None:
    target = [
        np.array([[-0.08515], [0.04335]]),
        np.array([[0.08515], [0.04335]]),
        np.array([[0.08515], [-0.04335]]),
        np.array([[-0.08515], [-0.04335]]),
    ]
    for v_target, v in zip(target, box_geometry.vertices):
        assert np.allclose(v_target, v)


def test_faces(box_geometry: Box2d) -> None:
    targets = [
        Hyperplane(a=np.array([[-0.0], [1.0]]), b=np.array([[0.04335]])),
        Hyperplane(a=np.array([[1.0], [0.0]]), b=np.array([[0.08515]])),
        Hyperplane(a=np.array([[-0.0], [-1.0]]), b=np.array([[0.04335]])),
        Hyperplane(a=np.array([[-1.0], [0.0]]), b=np.array([[0.08515]])),
    ]
    for face, target in zip(box_geometry.faces, targets):
        assert face == target


def test_normal_vecs(box_geometry: Box2d) -> None:
    targets = [
        np.array([[0], [-1]]),
        np.array([[-1], [0]]),
        np.array([[0], [1]]),
        np.array([[1], [0]]),
    ]
    normal_vecs = box_geometry.normal_vecs
    for n, t in zip(normal_vecs, targets):
        assert np.allclose(n, t)


def test_tangent_vecs(box_geometry: Box2d) -> None:
    targets = [
        np.array([[1], [0]]),
        np.array([[0], [-1]]),
        np.array([[-1], [0]]),
        np.array([[0], [1]]),
    ]
    tangent_vecs = box_geometry.tangent_vecs
    for n, t in zip(tangent_vecs, targets):
        assert np.allclose(n, t)


# NOTE: These test general functionality in AbstractCollisionGeometry and do
# not need to be implemented for all geometries
def test_get_lam_from_p_BP_by_projection() -> None:
    l = 1.0
    box = Box2d(width=l, height=l)  # square box
    loc = PolytopeContactLocation(ContactLocation.FACE, 0)  # top face

    p_BP = np.array([0, l / 2]).reshape((2, 1))
    lam_target = 0.5
    lam = box.get_lam_from_p_BP_by_projection(p_BP, loc)
    assert lam == lam_target

    p_BP = np.array([l / 2, l / 2]).reshape((2, 1))
    lam_target = 0.0
    lam = box.get_lam_from_p_BP_by_projection(p_BP, loc)
    assert lam == lam_target

    p_BP = np.array([-l / 2, l / 2]).reshape((2, 1))
    lam_target = 1.0
    lam = box.get_lam_from_p_BP_by_projection(p_BP, loc)
    assert lam == lam_target

    r = 0.1
    p_BP = np.array([l / 2, l / 2 + r]).reshape((2, 1))
    lam_target = 0.0
    lam = box.get_lam_from_p_BP_by_projection(p_BP, loc)
    assert lam == lam_target

    p_BP = np.array([-l / 2, l / 2 + r]).reshape((2, 1))
    lam_target = 1.0
    lam = box.get_lam_from_p_BP_by_projection(p_BP, loc)
    assert lam == lam_target

    # Should project the right value at any height
    p_BP = np.array([-l / 2, 99]).reshape((2, 1))
    lam_target = 1.0
    lam = box.get_lam_from_p_BP_by_projection(p_BP, loc)
    assert lam == lam_target

    # Should project the right value at any height
    p_BP = np.array([0, 99]).reshape((2, 1))
    lam_target = 0.5
    lam = box.get_lam_from_p_BP_by_projection(p_BP, loc)
    assert lam == lam_target

    # Should project the right value at any height
    p_BP = np.array([0, 0]).reshape((2, 1))
    lam_target = 0.5
    lam = box.get_lam_from_p_BP_by_projection(p_BP, loc)
    assert lam == lam_target


def test_get_p_Bc_from_lam() -> None:
    l = 1.0
    box = Box2d(width=l, height=l)  # square box
    loc = PolytopeContactLocation(ContactLocation.FACE, 0)  # top face

    lam = 0.5
    p_BP = box.get_p_Bc_from_lam(lam, loc)
    assert p_BP.shape == (2, 1)
    p_BP_target = np.array([0, l / 2])
    assert np.allclose(p_BP.flatten(), p_BP_target)

    # confusingly, lam = 0 gives the second vertex and
    # lam = 1 gives the first vertex on the contact face
    lam = 0.0
    p_BP = box.get_p_Bc_from_lam(lam, loc)
    p_BP_target = np.array([l / 2, l / 2])
    assert np.allclose(p_BP.flatten(), p_BP_target)

    lam = 1.0
    p_BP = box.get_p_Bc_from_lam(lam, loc)
    p_BP_target = np.array([-l / 2, l / 2])
    assert np.allclose(p_BP.flatten(), p_BP_target)


def test_get_p_BP_from_lam() -> None:
    l = 1.0
    box = Box2d(width=l, height=l)  # square box
    r = 0.1
    loc = PolytopeContactLocation(ContactLocation.FACE, 0)  # top face

    lam = 0.5
    p_BP = box.get_p_BP_from_lam(lam, loc, radius=0)
    assert p_BP.shape == (2, 1)
    p_BP_target = np.array([0, l / 2])
    assert np.allclose(p_BP.flatten(), p_BP_target)

    lam = 0.5
    p_BP = box.get_p_BP_from_lam(lam, loc, r)
    p_BP_target = np.array([0, l / 2 + r])
    assert np.allclose(p_BP.flatten(), p_BP_target)

    # confusingly, lam = 0 gives the second vertex and
    # lam = 1 gives the first vertex on the contact face
    lam = 0.0
    p_BP = box.get_p_BP_from_lam(lam, loc, r)
    p_BP_target = np.array([l / 2, l / 2 + r])
    assert np.allclose(p_BP.flatten(), p_BP_target)

    lam = 1.0
    p_BP = box.get_p_BP_from_lam(lam, loc, r)
    p_BP_target = np.array([-l / 2, l / 2 + r])
    assert np.allclose(p_BP.flatten(), p_BP_target)


# TODO: Complete test coverage
# def test_get_norm_and_tang_vecs(box_geometry: Box2d) -> None:
#     ...
#
# def test_get_neighbouring_vertices(box_geometry: Box2d) -> None:
#     ...
#
# def test_get_proximate_vertices_from_location(box_geometry: Box2d) -> None:
#     ...


def test_get_signed_distance() -> None:
    box = Box2d(width=0.3, height=0.2)

    # left, right, top, bottom
    assert np.isclose(box.get_signed_distance(np.array([-0.2, 0])), 0.05)
    assert np.isclose(box.get_signed_distance(np.array([0.2, 0])), 0.05)
    assert np.isclose(box.get_signed_distance(np.array([0, 0.2])), 0.1)
    assert np.isclose(box.get_signed_distance(np.array([0, -0.2])), 0.1)

    # corners
    target_dist = np.sqrt((0.2 - box.width / 2) ** 2 + (0.2 - box.height / 2) ** 2)
    assert np.isclose(box.get_signed_distance(np.array([-0.2, -0.2])), target_dist)
    assert np.isclose(box.get_signed_distance(np.array([-0.2, 0.2])), target_dist)
    assert np.isclose(box.get_signed_distance(np.array([0.2, -0.2])), target_dist)
    assert np.isclose(box.get_signed_distance(np.array([0.2, 0.2])), target_dist)

    # inside left
    assert np.isclose(box.get_signed_distance(np.array([-0.1, 0])), -0.05)
    # inside right
    assert np.isclose(box.get_signed_distance(np.array([0.1, 0])), -0.05)
    # inside top
    assert np.isclose(box.get_signed_distance(np.array([0, 0.07])), -0.03)
    # inside bottom
    assert np.isclose(box.get_signed_distance(np.array([0, -0.07])), -0.03)


def test_get_jacobian() -> None:
    box = Box2d(width=0.3, height=0.2)

    force_comps = np.array([0.1, 0.05]).reshape((2, 1))

    def _get_f(x, y):
        J = box.get_contact_jacobian(np.array([x, y]))
        f = J.T @ force_comps
        return f.flatten()

    # left
    f = _get_f(-box.width / 2, 0)
    assert np.isclose(f[0], force_comps[0])
    assert np.isclose(f[1], force_comps[1])
    target_torque = -box.width / 2 * force_comps[1].item()
    assert np.isclose(f[2], target_torque)

    # right
    f = _get_f(box.width / 2, 0)
    assert np.isclose(f[0], -force_comps[0])
    assert np.isclose(f[1], -force_comps[1])
    target_torque = -box.width / 2 * force_comps[1].item()
    assert np.isclose(f[2], target_torque)

    # top
    f = _get_f(0, box.height / 2)
    assert np.isclose(f[0], force_comps[1])
    assert np.isclose(f[1], -force_comps[0])
    target_torque = -box.height / 2 * force_comps[1].item()
    assert np.isclose(f[2], target_torque)

    # bottom
    f = _get_f(0, -box.height / 2)
    assert np.isclose(f[0], -force_comps[1])
    assert np.isclose(f[1], force_comps[0])
    target_torque = -box.height / 2 * force_comps[1].item()
    assert np.isclose(f[2], target_torque)
