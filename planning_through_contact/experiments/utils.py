from typing import Literal, Optional, Tuple, List

import numpy as np

from planning_through_contact.geometry.collision_geometry.box_2d import Box2d
from planning_through_contact.geometry.collision_geometry.t_pusher_2d import TPusher2d
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.rigid_body import RigidBody
from planning_through_contact.planning.planar.planar_plan_config import (
    ContactConfig,
    ContactCost,
    ContactCostType,
    NonCollisionCost,
    PlanarPlanConfig,
    PlanarPushingStartAndGoal,
    PlanarPushingWorkspace,
    PlanarSolverParams,
    SliderPusherSystemConfig,
)

from planning_through_contact.geometry.planar.non_collision import (
    check_finger_pose_in_contact_location,
)

from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    CollisionGeometry,
    ContactLocation,
    PolytopeContactLocation,
)

def get_box() -> RigidBody:
    mass = 0.1
    box_geometry = Box2d(width=0.1, height=0.1)
    slider = RigidBody("box", box_geometry, mass)
    return slider


def get_tee() -> RigidBody:
    mass = 0.1
    body = RigidBody("t_pusher", TPusher2d(), mass)
    return body


def get_sugar_box() -> RigidBody:
    mass = 0.1
    box_geometry = Box2d(width=0.106, height=0.185)
    slider = RigidBody("sugar_box", box_geometry, mass)
    return slider


def get_default_contact_cost() -> ContactCost:
    contact_cost = ContactCost(
        cost_type=ContactCostType.STANDARD,
        keypoint_arc_length=10.0,
        linear_arc_length=None,
        angular_arc_length=None,
        force_regularization=100000.0,  # NOTE: This is multiplied by 1e-4 because we have forces in other units in the optimization problem
        keypoint_velocity_regularization=100.0,
        ang_velocity_regularization=None,
        lin_velocity_regularization=None,
        trace=None,
        mode_transition_cost=None,
        time=1.0,
    )
    return contact_cost


def get_default_non_collision_cost() -> NonCollisionCost:
    non_collision_cost = NonCollisionCost(
        distance_to_object_socp=0.1,
        pusher_velocity_regularization=10.0,
        pusher_arc_length=10.0,
        time=None,
    )
    return non_collision_cost


def get_hardware_contact_cost() -> ContactCost:
    """
    A custom cost for hardware,
    which empically generates plans that respect robot velocity
    limits etc.
    """
    contact_cost = ContactCost(
        cost_type=ContactCostType.STANDARD,
        keypoint_arc_length=10.0,
        linear_arc_length=None,
        angular_arc_length=None,
        force_regularization=100000.0,
        keypoint_velocity_regularization=100.0,
        ang_velocity_regularization=None,
        lin_velocity_regularization=None,
        trace=None,
        mode_transition_cost=None,
        time=1.0,
    )
    return contact_cost


def get_hardware_non_collision_cost() -> NonCollisionCost:
    non_collision_cost = NonCollisionCost(
        distance_to_object_socp=0.25,
        pusher_velocity_regularization=10.0,
        pusher_arc_length=5.0,
        time=None,
    )
    return non_collision_cost


def get_default_plan_config(
    slider_type: Literal["box", "sugar_box", "tee"] = "box",
    pusher_radius: float = 0.015,
    time_contact: float = 2.0,
    time_non_collision: float = 4.0,
    workspace: Optional[PlanarPushingWorkspace] = None,
    hardware: bool = False,
) -> PlanarPlanConfig:
    if slider_type == "box":
        slider = get_box()
    elif slider_type == "sugar_box":
        slider = get_sugar_box()
    elif slider_type == "tee":
        slider = get_tee()
    else:
        raise NotImplementedError(f"Slider type {slider_type} not supported")

    if hardware:
        slider_pusher_config = SliderPusherSystemConfig(
            slider=slider,
            pusher_radius=pusher_radius,
            friction_coeff_slider_pusher=0.05,
            friction_coeff_table_slider=0.5,
            integration_constant=0.3,
        )

        contact_cost = get_hardware_contact_cost()
        non_collision_cost = get_hardware_non_collision_cost()
        lam_buffer = 0.25
        contact_config = ContactConfig(
            cost=contact_cost, lam_min=lam_buffer, lam_max=1 - lam_buffer
        )
        time_contact = 5.0
        time_non_collision = 2.0

        num_knot_points_non_collision = 5
        num_knot_points_contact = 3
    else:
        slider_pusher_config = SliderPusherSystemConfig(
            slider=slider,
            pusher_radius=pusher_radius,
            friction_coeff_slider_pusher=0.1,
            friction_coeff_table_slider=0.5,
            integration_constant=0.3,
        )
        contact_cost = get_default_contact_cost()
        non_collision_cost = get_default_non_collision_cost()
        lam_buffer = 0.0
        contact_config = ContactConfig(
            cost=contact_cost, lam_min=lam_buffer, lam_max=1 - lam_buffer
        )

        time_contact = 4.0
        time_non_collision = 2.0

        num_knot_points_non_collision = 3
        num_knot_points_contact = 3

    plan_cfg = PlanarPlanConfig(
        dynamics_config=slider_pusher_config,
        num_knot_points_contact=num_knot_points_contact,
        num_knot_points_non_collision=num_knot_points_non_collision,
        use_band_sparsity=True,
        contact_config=contact_config,
        non_collision_cost=non_collision_cost,
        continuity_on_pusher_velocity=True,
        allow_teleportation=False,
        time_in_contact=time_contact,
        time_non_collision=time_non_collision,
        workspace=workspace,
    )

    return plan_cfg


def get_default_solver_params(
    debug: bool = False, clarabel: bool = False
) -> PlanarSolverParams:
    solver_params = PlanarSolverParams(
        measure_solve_time=debug,
        rounding_steps=100,
        print_flows=False,
        solver="mosek" if not clarabel else "clarabel",
        print_solver_output=debug,
        save_solver_output=False,
        print_rounding_details=debug,
        print_path=False,
        print_cost=debug,
        assert_result=False,
        assert_nan_values=True,
        nonl_round_major_feas_tol=1e-5,
        nonl_round_minor_feas_tol=1e-5,
        nonl_round_opt_tol=1e-5,
    )
    return solver_params


def sample_random_plan(
    x_and_y_limits: Tuple[float, float, float, float] = (-0.5, 0.5, -0.5, 0.5),
    slider_target_pose: Optional[PlanarPose] = None,
):
    x_min, x_max, y_min, y_max = x_and_y_limits

    # Default target is origin
    if slider_target_pose is None:
        slider_target_pose = PlanarPose(0, 0, 0)

    # Draw random initial pose for slider
    x_initial = np.random.uniform(x_min, x_max)
    y_initial = np.random.uniform(y_min, y_max)
    th_initial = np.random.uniform(-np.pi + 0.1, np.pi - 0.1)

    slider_initial_pose = PlanarPose(x_initial, y_initial, th_initial)

    # Fix pusher pose to upper right corner, outside of where the
    # slider will be
    BUFFER = 0.5  # This is just a hardcoded distance number
    pusher_pose = PlanarPose(x_max + BUFFER, y_max + BUFFER, 0)

    plan = PlanarPushingStartAndGoal(
        slider_initial_pose, slider_target_pose, pusher_pose, pusher_pose
    )
    return plan

# Util functions for getting initial slider poses

def get_slider_start_poses(
    seed: int,
    num_plans: int,
    workspace: PlanarPushingWorkspace,
    config: PlanarPlanConfig,
    pusher_pose: PlanarPose,
    limit_rotations: bool = True,  # Use this to start with
) -> List[PlanarPushingStartAndGoal]:
    # We want the plans to always be the same
    np.random.seed(seed)
    slider = config.slider_geometry
    slider_initial_poses = []
    for _ in range(num_plans):
        slider_initial_pose = _get_slider_pose_within_workspace(
            workspace, slider, pusher_pose, config, limit_rotations
        )
        slider_initial_poses.append(slider_initial_pose)

    return slider_initial_poses

def _get_slider_pose_within_workspace(
    workspace: PlanarPushingWorkspace,
    slider: CollisionGeometry,
    pusher_pose: PlanarPose,
    config: PlanarPlanConfig,
    limit_rotations: bool = False,
    enforce_entire_slider_within_workspace: bool = True,
) -> PlanarPose:
    valid_pose = False

    slider_pose = None
    while not valid_pose:
        x_initial = np.random.uniform(workspace.slider.x_min, workspace.slider.x_max)
        y_initial = np.random.uniform(workspace.slider.y_min, workspace.slider.y_max)
        EPS = 0.01
        if limit_rotations:
            # th_initial = np.random.uniform(-np.pi / 2 + EPS, np.pi / 2 - EPS)
            th_initial = np.random.uniform(-np.pi / 4 + EPS, np.pi / 4 - EPS)
        else:
            th_initial = np.random.uniform(-np.pi + EPS, np.pi - EPS)

        slider_pose = PlanarPose(x_initial, y_initial, th_initial)

        collides_with_pusher = _check_collision(pusher_pose, slider_pose, config)
        within_workspace = _slider_within_workspace(workspace, slider_pose, slider)

        if enforce_entire_slider_within_workspace:
            valid_pose = within_workspace and not collides_with_pusher
        else:
            valid_pose = not collides_with_pusher

    assert slider_pose is not None  # fix LSP errors

    return slider_pose

# TODO: refactor
def _check_collision(
    pusher_pose_world: PlanarPose,
    slider_pose_world: PlanarPose,
    config: PlanarPlanConfig,
) -> bool:
    p_WP = pusher_pose_world.pos()
    R_WB = slider_pose_world.two_d_rot_matrix()
    p_WB = slider_pose_world.pos()

    # We need to compute the pusher pos in the frame of the slider
    p_BP = R_WB.T @ (p_WP - p_WB)
    pusher_pose_body = PlanarPose(p_BP[0, 0], p_BP[1, 0], 0)

    # we always add all non-collision modes, even when we don't add all contact modes
    # (think of maneuvering around the object etc)
    locations = [
        PolytopeContactLocation(ContactLocation.FACE, idx)
        for idx in range(config.slider_geometry.num_collision_free_regions)
    ]
    matching_locs = [
        loc
        for loc in locations
        if check_finger_pose_in_contact_location(pusher_pose_body, loc, config)
    ]
    if len(matching_locs) == 0:
        return True
    else:
        return False
    
def _slider_within_workspace(
    workspace: PlanarPushingWorkspace, pose: PlanarPose, slider: CollisionGeometry
) -> bool:
    """
    Checks whether the entire slider is within the workspace
    """
    R_WB = pose.two_d_rot_matrix()
    p_WB = pose.pos()

    p_Wv_s = [
        slider.get_p_Wv_i(vertex_idx, R_WB, p_WB).flatten()
        for vertex_idx in range(len(slider.vertices))
    ]

    lb, ub = workspace.slider.bounds
    vertices_within_workspace: bool = np.all([v <= ub for v in p_Wv_s]) and np.all(
        [v >= lb for v in p_Wv_s]
    )
    return vertices_within_workspace
