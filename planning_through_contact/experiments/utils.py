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
    PlanarPushingWorkspace,
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
        force_regularization=1.0,
        keypoint_velocity_regularization=None,
        ang_velocity_regularization=1.0,
        lin_velocity_regularization=1.0,
        trace=None,
        mode_transition_cost=None,
        time=None,
    )
    return contact_cost


def get_default_non_collision_cost() -> NonCollisionCost:
    non_collision_cost = NonCollisionCost(
        distance_to_object_socp=0.001,  # this sometimes cause numerical problems
        pusher_velocity_regularization=0.1,
        pusher_arc_length=0.1,
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
        keypoint_arc_length=5.0,
        linear_arc_length=None,
        angular_arc_length=None,
        force_regularization=1.0,
        keypoint_velocity_regularization=None,
        ang_velocity_regularization=10.0,
        lin_velocity_regularization=2.0,
        trace=None,
        mode_transition_cost=None,
        time=0.35,
    )
    return contact_cost


def get_hardware_non_collision_cost() -> NonCollisionCost:
    non_collision_cost = NonCollisionCost(
        # distance_to_object_quadratic=0.15,
        # distance_to_object_quadratic_preferred_distance=0.075,
        # distance_to_object_socp=None,
        distance_to_object_socp=0.0001,  # this sometimes cause numerical problems
        pusher_velocity_regularization=0.002,
        pusher_arc_length=0.004,
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
        time_contact = 2.0
        time_non_collision = 4.0
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

    plan_cfg = PlanarPlanConfig(
        dynamics_config=slider_pusher_config,
        num_knot_points_contact=3,
        num_knot_points_non_collision=3,
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
        rounding_steps=300,
        print_flows=False,
        solver="mosek" if not clarabel else "clarabel",
        print_solver_output=debug,
        save_solver_output=False,
        print_path=False,
        print_cost=debug,
        assert_result=False,
        assert_nan_values=True,
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
