import numpy as np
import pytest

from planning_through_contact.geometry.collision_geometry.box_2d import Box2d
from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    ContactLocation,
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


# TODO: Complete test coverage
# def test_get_norm_and_tang_vecs(box_geometry: Box2d) -> None:
#     ...
#
# def test_get_neighbouring_vertices(box_geometry: Box2d) -> None:
#     ...
#
# def test_get_proximate_vertices_from_location(box_geometry: Box2d) -> None:
#     ...
