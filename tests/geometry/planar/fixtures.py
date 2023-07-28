import pydrake.geometry.optimization as opt
import pytest
from pydrake.solvers import CommonSolverOption, MathematicalProgram, SolverOptions

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


# @pytest.fixture
def box_geometry() -> Box2d:
    return Box2d(width=0.3, height=0.3)


# @pytest.fixture
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


# @pytest.fixture
def gcs_options() -> opt.GraphOfConvexSetsOptions:
    options = opt.GraphOfConvexSetsOptions()
    options.solver_options = SolverOptions()
    options.convex_relaxation = True
    options.preprocessing = True
    options.max_rounded_paths = 1

    DEBUG = False
    if DEBUG:
        options.solver_options.SetOption(CommonSolverOption.kPrintToConsole, 1)  # type: ignore

    return options
