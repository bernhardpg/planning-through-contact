from typing import Tuple

import pytest
from pydrake.solvers import MathematicalProgram

from planning_through_contact.geometry.collision_geometry.box_2d import Box2d
from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    ContactLocation,
    PolytopeContactLocation,
)
from planning_through_contact.geometry.planar.abstract_mode import PlanarPlanSpecs
from planning_through_contact.geometry.planar.face_contact import (
    FaceContactMode,
    FaceContactVariables,
)
from planning_through_contact.geometry.planar.non_collision import (
    NonCollisionMode,
    NonCollisionVariables,
)
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.rigid_body import RigidBody


@pytest.fixture
def box_geometry() -> Box2d:
    return Box2d(width=0.3, height=0.3)


@pytest.fixture
def rigid_body_box(box_geometry: Box2d) -> RigidBody:
    mass = 0.3
    box = RigidBody("box", box_geometry, mass)
    return box


@pytest.fixture
def face_contact_vars(box_geometry: Box2d) -> FaceContactVariables:
    prog = MathematicalProgram()
    contact_location = PolytopeContactLocation(ContactLocation.FACE, 3)

    num_knot_points = 4
    time_in_contact = 2

    vars = FaceContactVariables.from_prog(
        prog,
        box_geometry,
        contact_location,
        num_knot_points,
        time_in_contact,
    )
    return vars


@pytest.fixture
def non_collision_vars() -> NonCollisionVariables:
    prog = MathematicalProgram()

    num_knot_points = 2
    time_in_contact = 2

    vars = NonCollisionVariables.from_prog(
        prog,
        num_knot_points,
        time_in_contact,
    )
    return vars


@pytest.fixture
def non_collision_mode(rigid_body_box: RigidBody) -> NonCollisionMode:
    contact_location = PolytopeContactLocation(ContactLocation.FACE, 3)
    specs = PlanarPlanSpecs()
    mode = NonCollisionMode.create_from_plan_spec(
        contact_location, specs, rigid_body_box
    )

    return mode


@pytest.fixture
def face_contact_mode(rigid_body_box: RigidBody) -> FaceContactMode:
    contact_location = PolytopeContactLocation(ContactLocation.FACE, 3)
    specs = PlanarPlanSpecs()
    mode = FaceContactMode.create_from_plan_spec(
        contact_location, specs, rigid_body_box
    )
    return mode


@pytest.fixture
def initial_and_final_non_collision_mode(
    rigid_body_box: RigidBody,
) -> Tuple[NonCollisionMode, NonCollisionMode]:
    plan_specs = PlanarPlanSpecs()

    contact_location_start = PolytopeContactLocation(ContactLocation.FACE, 3)
    contact_location_end = PolytopeContactLocation(ContactLocation.FACE, 0)

    source_mode = NonCollisionMode.create_from_plan_spec(
        contact_location_start, plan_specs, rigid_body_box, "source"
    )
    target_mode = NonCollisionMode.create_from_plan_spec(
        contact_location_end, plan_specs, rigid_body_box, "target"
    )

    slider_pose = PlanarPose(0.3, 0, 0)
    source_mode.set_slider_pose(slider_pose)
    target_mode.set_slider_pose(slider_pose)

    finger_initial_pose = PlanarPose(-0.2, 0, 0)
    source_mode.set_finger_initial_pos(finger_initial_pose.pos())
    finger_final_pose = PlanarPose(0.05, 0.5, 0)
    target_mode.set_finger_final_pos(finger_final_pose.pos())

    return source_mode, target_mode