import pytest

from planning_through_contact.geometry.collision_geometry.box_2d import Box2d
from planning_through_contact.geometry.collision_geometry.box_group_2d import BoxGroup2d
from planning_through_contact.geometry.collision_geometry.t_pusher_2d import TPusher
from planning_through_contact.geometry.planar.planar_pose import PlanarPose


def test_visualize() -> None:
    # 1. Define T-pusher as a collision geometry (should just be two rectangles)
    # 2. Add geometry to the scene graph
    # 3. Connect the trajectory feeder to the geometry system

    slider = TPusher()

    breakpoint()
