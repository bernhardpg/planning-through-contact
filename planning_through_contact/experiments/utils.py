from typing import Literal

from planning_through_contact.geometry.collision_geometry.box_2d import Box2d
from planning_through_contact.geometry.collision_geometry.t_pusher_2d import TPusher2d
from planning_through_contact.geometry.rigid_body import RigidBody
from planning_through_contact.planning.planar.planar_plan_config import (
    ContactConfig,
    ContactCost,
    ContactCostType,
    NonCollisionCost,
    PlanarPlanConfig,
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
        integration_constant=0.6,
    )

    contact_cost = get_default_contact_cost()
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
