import pytest

from geometry.collision_geometry.box_2d import Box2d
from geometry.rigid_body import RigidBody
from tests.geometry.collision_geometry.test_box2d import box_geometry


# @pytest.fixture
def rigid_body_box(box_geometry: Box2d) -> RigidBody:
    box_mass = 0.514
    body = RigidBody("box", box_geometry, box_mass)
    return body
