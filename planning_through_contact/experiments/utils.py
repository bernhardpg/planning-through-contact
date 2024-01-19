from typing import Literal, Optional, Tuple

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
    PlanarSolverParams,
    SliderPusherSystemConfig,
)


def get_box() -> RigidBody:
    mass = 0.1
    box_geometry = Box2d(width=0.07, height=0.07)
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
        keypoint_arc_length=1.0,
        linear_arc_length=None,
        angular_arc_length=None,
        force_regularization=0.3,
        keypoint_velocity_regularization=None,
        ang_velocity_regularization=1.0,
        lin_velocity_regularization=0.1,
        trace=None,
    )
    return contact_cost


def get_default_non_collision_cost() -> NonCollisionCost:
    non_collision_cost = NonCollisionCost(
        distance_to_object_quadratic=None,
        distance_to_object_socp=0.001,
        pusher_velocity_regularization=0.005,
        pusher_arc_length=0.005,
    )
    return non_collision_cost


def get_default_plan_config(
    slider_type: Literal["box", "sugar_box", "tee"] = "box",
    pusher_radius: float = 0.035,
    integration_constant: float = 0.6,
    arc_length_weight: Optional[float] = None,
) -> PlanarPlanConfig:
    if slider_type == "box":
        slider = get_box()
    elif slider_type == "sugar_box":
        slider = get_sugar_box()
    elif slider_type == "tee":
        slider = get_tee()
    else:
        raise NotImplementedError(f"Slider type {slider_type} not supported")

    # Define slider-pusher system
    slider_pusher_config = SliderPusherSystemConfig(
        slider=slider,
        pusher_radius=pusher_radius,
        friction_coeff_slider_pusher=0.4,
        friction_coeff_table_slider=0.5,
        integration_constant=integration_constant,
    )

    contact_cost = get_default_contact_cost()
    if arc_length_weight is not None:
        contact_cost.keypoint_arc_length = arc_length_weight

    non_collision_cost = get_default_non_collision_cost()

    contact_config = ContactConfig(
        cost=contact_cost,
        lam_min=0.1,
        lam_max=0.9,
    )

    plan_cfg = PlanarPlanConfig(
        dynamics_config=slider_pusher_config,
        num_knot_points_contact=3,
        num_knot_points_non_collision=3,
        use_band_sparsity=True,
        contact_config=contact_config,
        non_collision_cost=non_collision_cost,
        continuity_on_pusher_velocity=True,
        allow_teleportation=False,
    )

    return plan_cfg


def get_default_solver_params() -> PlanarSolverParams:
    solver_params = PlanarSolverParams(
        measure_solve_time=False,
        gcs_max_rounded_paths=20,
        print_flows=False,
        solver="mosek",
        print_solver_output=False,
        save_solver_output=False,
        print_path=False,
        print_cost=False,
        assert_result=False,
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
