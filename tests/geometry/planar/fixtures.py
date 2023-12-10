import numpy as np
import pydrake.geometry.optimization as opt
import pytest
from _pytest.fixtures import FixtureRequest
from pydrake.solvers import (
    CommonSolverOption,
    MathematicalProgram,
    MosekSolver,
    SolverOptions,
)

from planning_through_contact.convex_relaxation.band_sparse_semidefinite_relaxation import (
    BandSparseSemidefiniteRelaxation,
)
from planning_through_contact.geometry.collision_geometry.box_2d import Box2d
from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    ContactLocation,
    PolytopeContactLocation,
)
from planning_through_contact.geometry.collision_geometry.t_pusher_2d import TPusher2d
from planning_through_contact.geometry.planar.abstract_mode import PlanarPlanConfig
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
from planning_through_contact.planning.planar.planar_plan_config import (
    BoxWorkspace,
    ContactCostType,
    SliderPusherSystemConfig,
)
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
def dynamics_config(rigid_body_box: RigidBody) -> SliderPusherSystemConfig:
    cfg = SliderPusherSystemConfig(slider=rigid_body_box, pusher_radius=0.0)
    return cfg


@pytest.fixture
def plan_config(dynamics_config: SliderPusherSystemConfig) -> PlanarPlanConfig:
    cfg = PlanarPlanConfig(
        dynamics_config=dynamics_config,
        use_approx_exponential_map=False,
        use_band_sparsity=False,
        avoidance_cost="quadratic",  # TODO: Tests should be updated to use socp cost
    )
    cfg.contact_cost.type = ContactCostType.SQ_VELOCITIES
    return cfg


@pytest.fixture
def t_pusher() -> RigidBody:
    mass = 0.3
    return RigidBody("box", TPusher2d(), mass)


@pytest.fixture
def face_contact_vars(box_geometry: Box2d) -> FaceContactVariables:
    num_knot_points = 4
    prog = BandSparseSemidefiniteRelaxation(num_groups=num_knot_points)
    contact_location = PolytopeContactLocation(ContactLocation.FACE, 3)

    time_in_contact = 2

    vars = FaceContactVariables.from_prog(
        prog,
        box_geometry,
        contact_location,
        num_knot_points,
        time_in_contact,
        pusher_radius=0,
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
        pusher_radius=0,
    )
    return vars


@pytest.fixture
def non_collision_mode(plan_config: PlanarPlanConfig) -> NonCollisionMode:
    contact_location = PolytopeContactLocation(ContactLocation.FACE, 3)
    mode = NonCollisionMode.create_from_plan_spec(contact_location, plan_config)

    return mode


@pytest.fixture
def face_contact_mode(
    plan_config: PlanarPlanConfig, t_pusher: RigidBody, request: FixtureRequest
) -> FaceContactMode:
    if not hasattr(request, "param"):
        request.param = {}  # Make the fixture work without params

    if request.param.get("body") == "t_pusher":
        plan_config.dynamics_config.slider = t_pusher

    plan_config.contact_cost.type = request.param.get(
        "contact_cost", ContactCostType.SQ_VELOCITIES
    )

    face_idx = request.param.get("face_idx", 3)
    plan_config.use_eq_elimination = request.param.get("use_eq_elimination", False)

    contact_location = PolytopeContactLocation(ContactLocation.FACE, face_idx)
    mode = FaceContactMode.create_from_plan_spec(
        contact_location,
        plan_config,
    )
    return mode


@pytest.fixture
def gcs_options() -> opt.GraphOfConvexSetsOptions:
    options = opt.GraphOfConvexSetsOptions()
    options.solver_options = SolverOptions()
    options.convex_relaxation = True
    options.preprocessing = True
    options.max_rounded_paths = 1
    options.solver = MosekSolver()

    DEBUG = False
    if DEBUG:
        options.solver_options.SetOption(CommonSolverOption.kPrintToConsole, 1)  # type: ignore

    return options


@pytest.fixture
def subgraph(
    plan_config: PlanarPlanConfig, request: FixtureRequest
) -> NonCollisionSubGraph:
    num_knot_points = 4 if request.param["avoid_object"] else 2
    plan_config.num_knot_points_non_collision = num_knot_points
    plan_config.avoid_object = request.param.get("avoid_object", False)
    plan_config.avoidance_cost = request.param.get("avoidance_cost_type", "quadratic")

    gcs = opt.GraphOfConvexSets()

    subgraph = NonCollisionSubGraph.create_with_gcs(
        gcs,
        plan_config,
        "Subgraph_TEST",
    )

    if request.param["boundary_conds"]:
        slider_pose = PlanarPose(0.3, 0.3, 0)
        finger_initial_pose = request.param["finger_initial"]
        finger_final_pose = request.param["finger_final"]

        subgraph.set_initial_poses(finger_initial_pose, slider_pose)
        subgraph.set_final_poses(finger_final_pose, slider_pose)

    return subgraph


@pytest.fixture
def planner(
    plan_config: PlanarPlanConfig, t_pusher: RigidBody, request: FixtureRequest
) -> PlanarPushingPlanner:
    plan_config.workspace.slider = BoxWorkspace(
        width=2, height=2, center=np.array([0.0, 0.35])
    )
    plan_config.workspace.pusher = BoxWorkspace(width=2, height=2)

    body_to_use = request.param.get("body", "rigid_body_box")
    if body_to_use == "t_pusher":
        plan_config.dynamics_config.slider = t_pusher

    if request.param.get("partial"):
        contact_locations = plan_config.slider_geometry.contact_locations[0:2]
    else:
        contact_locations = plan_config.slider_geometry.contact_locations

    if request.param.get("avoid_object"):
        plan_config.num_knot_points_non_collision = 4
        plan_config.avoid_object = True
    else:
        plan_config.avoid_object = False

    plan_config.dynamics_config.pusher_radius = 0.015
    plan_config.use_band_sparsity = request.param.get("use_band_sparsity", False)

    plan_config.avoid_object = request.param.get("avoid_object", False)
    plan_config.allow_teleportation = request.param.get("allow_teleportation", False)
    plan_config.penalize_mode_transitions = request.param.get(
        "penalize_mode_transition", False
    )
    plan_config.avoidance_cost = request.param.get("avoidance_cost_type", "quadratic")
    plan_config.use_eq_elimination = request.param.get("use_eq_elimination", False)

    planner = PlanarPushingPlanner(
        plan_config,
        contact_locations=contact_locations,
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

    planner.formulate_problem()
    return planner
