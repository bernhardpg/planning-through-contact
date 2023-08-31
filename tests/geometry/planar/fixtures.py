import pydrake.geometry.optimization as opt
import pytest
from _pytest.fixtures import FixtureRequest
from pydrake.solvers import CommonSolverOption, MathematicalProgram, SolverOptions

from planning_through_contact.geometry.collision_geometry.box_2d import Box2d
from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    ContactLocation,
    PolytopeContactLocation,
)
from planning_through_contact.geometry.collision_geometry.t_pusher_2d import TPusher2d
from planning_through_contact.geometry.planar.abstract_mode import PlanarPlanSpecs
from planning_through_contact.geometry.planar.face_contact import (
    FaceContactMode,
    FaceContactVariables,
)
from planning_through_contact.geometry.planar.non_collision import (
    NonCollisionMode,
    NonCollisionVariables,
)
from planning_through_contact.geometry.planar.non_collision_subgraph import (
    NonCollisionSubGraph,
)
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.rigid_body import RigidBody
from planning_through_contact.planning.planar.planar_pushing_planner import (
    PlanarPushingPlanner,
)


@pytest.fixture
def box_geometry() -> Box2d:
    return Box2d(width=0.3, height=0.3)


@pytest.fixture
def rigid_body_box(box_geometry: Box2d) -> RigidBody:
    mass = 0.3
    box = RigidBody("box", box_geometry, mass)
    return box


@pytest.fixture
def t_pusher() -> RigidBody:
    mass = 0.3
    return RigidBody("box", TPusher2d(), mass)


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
        PolytopeContactLocation(ContactLocation.FACE, 0),
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
def face_contact_mode(
    rigid_body_box: RigidBody, t_pusher: RigidBody, request: FixtureRequest
) -> FaceContactMode:
    rigid_body = rigid_body_box

    if not hasattr(request, "param"):
        request.param = {}  # Make the fixture work without params

    if request.param.get("body") == "t_pusher":
        rigid_body = t_pusher

    face_idx = request.param.get("face_idx", 3)

    contact_location = PolytopeContactLocation(ContactLocation.FACE, face_idx)
    specs = PlanarPlanSpecs()
    mode = FaceContactMode.create_from_plan_spec(
        contact_location,
        specs,
        rigid_body,
        request.param.get("use_eq_elimination", False),
    )
    return mode


@pytest.fixture
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


@pytest.fixture
def subgraph(
    rigid_body_box: RigidBody, request: FixtureRequest
) -> NonCollisionSubGraph:
    num_knot_points = 4 if request.param["avoid_object"] else 2

    plan_specs = PlanarPlanSpecs(num_knot_points_non_collision=num_knot_points)
    gcs = opt.GraphOfConvexSets()

    subgraph = NonCollisionSubGraph.create_with_gcs(
        gcs,
        rigid_body_box,
        plan_specs,
        "Subgraph_TEST",
        avoid_object=request.param["avoid_object"],
    )

    if request.param["boundary_conds"]:
        slider_pose = PlanarPose(0.3, 0, 0)
        finger_initial_pose = request.param["finger_initial"]
        finger_final_pose = request.param["finger_final"]

        subgraph.set_initial_poses(finger_initial_pose, slider_pose)
        subgraph.set_final_poses(finger_final_pose, slider_pose)

    return subgraph


@pytest.fixture
def planner(rigid_body_box: RigidBody, request: FixtureRequest) -> PlanarPushingPlanner:
    body_to_use = request.param.get("body", "rigid_body_box")
    if body_to_use == "rigid_body_box":
        body = rigid_body_box
    elif body_to_use == "t_pusher":
        mass = 0.3
        body = RigidBody("t_pusher", TPusher2d(), mass)
    else:
        body = rigid_body_box

    if request.param.get("partial"):
        contact_locations = body.geometry.contact_locations[0:2]
    else:
        contact_locations = body.geometry.contact_locations

    if request.param.get("avoid_object"):
        specs = PlanarPlanSpecs(num_knot_points_non_collision=4)
    else:
        specs = PlanarPlanSpecs()

    avoid_object = request.param.get("avoid_object", False)
    allow_teleportation = request.param.get("allow_teleportation", False)
    penalize_mode_transition = request.param.get("penalize_mode_transition", False)
    avoidance_cost_type = request.param.get("avoidance_cost_type", "quadratic")
    use_eq_elimination = request.param.get("use_eq_elimination", False)
    use_redundant_dynamic_constraints = request.param.get(
        "use_redundant_dynamic_constraints", True
    )

    planner = PlanarPushingPlanner(
        body,
        specs,
        contact_locations=contact_locations,
        avoid_object=avoid_object,
        allow_teleportation=allow_teleportation,
        penalize_mode_transition=penalize_mode_transition,
        avoidance_cost_type=avoidance_cost_type,
        use_eq_elimination=use_eq_elimination,
        use_redundant_dynamic_constraints=use_redundant_dynamic_constraints,
    )

    if request.param.get("boundary_conds"):
        boundary_conds = request.param.get("boundary_conds")

        planner.set_initial_poses(
            boundary_conds["finger_initial_pose"],
            boundary_conds["box_initial_pose"],
        )
        planner.set_target_poses(
            boundary_conds["finger_target_pose"],
            boundary_conds["box_target_pose"],
        )

    return planner
