import numpy as np
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
from planning_through_contact.geometry.rigid_body import RigidBody


@pytest.fixture
def box_geometry() -> Box2d:
    return Box2d(width=0.3, height=0.3)


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
def face_contact_mode(box_geometry: Box2d) -> FaceContactMode:
    mass = 0.3
    box = RigidBody("box", box_geometry, mass)
    contact_location = PolytopeContactLocation(
        ContactLocation.FACE, 3
    )  # We use the same face as in the T-pusher demo to make it simple to write tests
    specs = PlanarPlanSpecs()
    mode = FaceContactMode.create_from_plan_spec(contact_location, specs, box)
    return mode
