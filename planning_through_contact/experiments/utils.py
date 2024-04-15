import os
from datetime import datetime
from typing import List, Literal, Optional, Tuple

import numpy as np

from planning_through_contact.experiments.ablation_study.planar_pushing_ablation import (
    run_ablation,
)
from planning_through_contact.geometry.collision_geometry.box_2d import Box2d
from planning_through_contact.geometry.collision_geometry.t_pusher_2d import TPusher2d
from planning_through_contact.geometry.rigid_body import RigidBody
from planning_through_contact.planning.planar.planar_plan_config import (
    BoxWorkspace,
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
from planning_through_contact.planning.planar.utils import (
    get_plan_start_and_goals_to_point,
)


def create_output_folder(
    output_dir: str, slider_type: str, traj_number: Optional[int]
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    folder_name = f"{output_dir}/run_{get_time_as_str()}_{slider_type}"
    if traj_number is not None:
        folder_name += f"_traj_{traj_number}"
    os.makedirs(folder_name, exist_ok=True)

    return folder_name


def get_time_as_str() -> str:
    current_time = datetime.now()
    # For example, YYYYMMDDHHMMSS format
    formatted_time = current_time.strftime("%Y%m%d%H%M%S")
    return formatted_time


def get_box() -> RigidBody:
    mass = 0.1
    box_geometry = Box2d(width=0.2, height=0.2)
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


def get_hardware_plans(
    hardware_seed: int, config: PlanarPlanConfig
) -> List[PlanarPushingStartAndGoal]:
    """
    Generates a collection of plans that can be run on our hardware setup with the Kuka Iiwa, with the right workspace
    and origin.
    """
    workspace = PlanarPushingWorkspace(
        slider=BoxWorkspace(
            width=0.35,
            height=0.5,
            center=np.array([0.575, 0.0]),
            buffer=0,
        ),
    )

    num_trajs = 30
    plans = get_plan_start_and_goals_to_point(
        hardware_seed,
        num_trajs,
        workspace,
        config,
        (0.575, -0.04285714),
        limit_rotations=False,
    )

    return plans


def get_default_experiment_plans(
    seed: int, num_trajs: int, config: PlanarPlanConfig, workspace_size: float = 0.6
) -> List[PlanarPushingStartAndGoal]:
    """
    Generates a collection of random initial configurations with the origin as the target
    configuration.
    """
    workspace = PlanarPushingWorkspace(
        slider=BoxWorkspace(
            width=workspace_size,
            height=workspace_size,
            center=np.array([0.0, 0.0]),
            buffer=0,
        ),
    )

    plans = get_plan_start_and_goals_to_point(
        seed,
        num_trajs,
        workspace,
        config,
        (0.0, 0.0),
        limit_rotations=False,
    )

    return plans


def run_ablation_with_default_config(
    slider_type: Literal["box", "sugar_box", "tee"],
    pusher_radius: float,
    integration_constant: float,
    num_experiments: int,
    arc_length_weight: Optional[float] = None,
    filename: Optional[str] = None,
) -> None:
    config = get_default_plan_config(
        slider_type, pusher_radius, integration_constant, arc_length_weight  # type: ignore
    )
    solver_params = get_default_solver_params()
    run_ablation(config, solver_params, num_experiments, filename)  # type: ignore


def get_baseline_comparison_costs() -> Tuple[ContactCost, NonCollisionCost]:
    contact_cost = get_default_contact_cost()
    non_collision_cost = get_default_non_collision_cost()

    return contact_cost, non_collision_cost


def get_baseline_comparison_configs(
    slider_type: str = "sugar_box",
) -> Tuple[PlanarPlanConfig, PlanarSolverParams]:
    config = get_default_plan_config()
    dt = 0.25
    config.num_knot_points_contact = 4
    config.time_in_contact = config.num_knot_points_contact * dt
    config.num_knot_points_non_collision = 3
    config.time_non_collision = config.num_knot_points_non_collision * dt

    assert config.dt_contact == config.dt_non_collision

    solver_params = get_default_solver_params()

    solver_params.nonl_rounding_save_solver_output = False
    solver_params.print_cost = False

    return config, solver_params
