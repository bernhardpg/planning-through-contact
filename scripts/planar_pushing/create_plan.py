import argparse
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import numpy as np

from planning_through_contact.geometry.collision_geometry.box_2d import Box2d
from planning_through_contact.geometry.collision_geometry.t_pusher_2d import TPusher2d
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.rigid_body import RigidBody
from planning_through_contact.planning.planar.planar_plan_config import (
    ContactConfig,
    ContactCostType,
    PlanarCostFunctionTerms,
    PlanarPlanConfig,
    PlanarPushingStartAndGoal,
    PlanarSolverParams,
    SliderPusherSystemConfig,
)
from planning_through_contact.planning.planar.planar_pushing_planner import (
    PlanarPushingPlanner,
)
from planning_through_contact.visualize.planar_pushing import (
    visualize_planar_pushing_start_and_goal,
    visualize_planar_pushing_trajectory,
)


def get_slider_box() -> RigidBody:
    mass = 0.1
    box_geometry = Box2d(width=0.15, height=0.15)
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


def get_predefined_plan(traj_number: int) -> PlanarPushingStartAndGoal:
    if traj_number == 1:
        slider_initial_pose = PlanarPose(x=0.55, y=0.0, theta=0.0)
        slider_target_pose = PlanarPose(x=0.65, y=0.0, theta=-0.5)
        pusher_initial_pose = PlanarPose(x=-0.2, y=-0.2, theta=0.0)
        pusher_target_pose = PlanarPose(x=-0.2, y=-0.2, theta=0.0)
    elif traj_number == 2:
        slider_initial_pose = PlanarPose(x=0.60, y=0.1, theta=-0.2)
        slider_target_pose = PlanarPose(x=0.70, y=-0.2, theta=0.5)
        pusher_initial_pose = PlanarPose(x=-0.2, y=-0.2, theta=0.0)
        pusher_target_pose = PlanarPose(x=-0.2, y=-0.2, theta=0.0)
    elif traj_number == 3:
        slider_initial_pose = PlanarPose(x=0.60, y=0.1, theta=-0.2)
        slider_target_pose = PlanarPose(x=0.70, y=0.4, theta=0.5)
        pusher_initial_pose = PlanarPose(x=-0.2, y=0.0, theta=0.0)
        pusher_target_pose = PlanarPose(x=-0.2, y=0.0, theta=0.0)
    elif traj_number == 4:
        slider_initial_pose = PlanarPose(x=0.60, y=0.1, theta=np.pi / 2 - 0.2)
        slider_target_pose = PlanarPose(x=0.75, y=-0.2, theta=np.pi / 2 + 0.4)
        pusher_initial_pose = PlanarPose(x=-0.2, y=0.0, theta=0.0)
        pusher_target_pose = PlanarPose(x=0.0, y=0.2, theta=0.0)
    elif traj_number == 5:
        slider_initial_pose = PlanarPose(x=0.60, y=0.1, theta=np.pi / 2)
        slider_target_pose = PlanarPose(x=0.70, y=-0.05, theta=np.pi / 2 + 0.4)
        pusher_initial_pose = PlanarPose(x=-0.2, y=0.0, theta=0.0)
        pusher_target_pose = PlanarPose(x=0.0, y=0.2, theta=0.0)
    elif traj_number == 6:
        slider_initial_pose = PlanarPose(x=0.60, y=0.0, theta=np.pi / 2)
        slider_target_pose = PlanarPose(x=0.65, y=-0.1, theta=np.pi / 2 + 0.3)
        pusher_initial_pose = PlanarPose(x=-0.2, y=0.0, theta=0.0)
        pusher_target_pose = PlanarPose(x=0.0, y=0.2, theta=0.0)
    elif traj_number == 7:
        slider_initial_pose = PlanarPose(x=0.0, y=0.0, theta=0.0)
        slider_target_pose = PlanarPose(x=0.3, y=-0.15, theta=-0.5)
        pusher_initial_pose = PlanarPose(x=0.0, y=0.2, theta=0.0)
        pusher_target_pose = PlanarPose(x=0.0, y=0.2, theta=0.0)
    elif traj_number == 8:
        slider_initial_pose = PlanarPose(x=0.0, y=0.0, theta=0.0)
        slider_target_pose = PlanarPose(x=-0.3, y=-0.15, theta=-0.5)
        pusher_initial_pose = PlanarPose(x=0.0, y=0.2, theta=0.0)
        pusher_target_pose = PlanarPose(x=0.0, y=0.2, theta=0.0)
    elif traj_number == 9:
        slider_initial_pose = PlanarPose(x=0.5, y=0.0, theta=1.0)
        slider_target_pose = PlanarPose(x=0.6, y=-0.15, theta=-0.5)
        pusher_initial_pose = PlanarPose(x=0.0, y=0.2, theta=0.0)
        pusher_target_pose = PlanarPose(x=0.0, y=0.2, theta=0.0)
    elif traj_number == 10:
        slider_initial_pose = PlanarPose(x=0.7, y=0.2, theta=0.3)
        slider_target_pose = PlanarPose(x=0.55, y=-0.15, theta=1.2)
        pusher_initial_pose = PlanarPose(x=0.0, y=0.2, theta=0.0)
        pusher_target_pose = PlanarPose(x=0.0, y=0.2, theta=0.0)
    elif traj_number == 11:
        # Rotate in place is a bit loose
        slider_initial_pose = PlanarPose(x=0.65, y=0.0, theta=0)
        slider_target_pose = PlanarPose(x=0.65, y=0.0, theta=np.pi / 2)
        pusher_initial_pose = PlanarPose(x=0.0, y=0.2, theta=0.0)
        pusher_target_pose = PlanarPose(x=0.0, y=0.2, theta=0.0)
    elif traj_number == 12:
        slider_initial_pose = PlanarPose(x=0.45, y=-0.1, theta=0.9)
        slider_target_pose = PlanarPose(x=0.45, y=0.1, theta=0.9)
        pusher_initial_pose = PlanarPose(x=-0.2, y=-0.2, theta=0.0)
        pusher_target_pose = PlanarPose(x=-0.2, y=-0.2, theta=0.0)
    elif traj_number == 13:
        slider_initial_pose = PlanarPose(x=0.45, y=-0.1, theta=0.9)
        slider_target_pose = PlanarPose(x=0.45, y=-0.1, theta=np.pi / 2)
        pusher_initial_pose = PlanarPose(x=-0.2, y=-0.2, theta=0.0)
        pusher_target_pose = PlanarPose(x=-0.2, y=-0.2, theta=0.0)
    else:
        raise NotImplementedError()

    return PlanarPushingStartAndGoal(
        slider_initial_pose, slider_target_pose, pusher_initial_pose, pusher_target_pose
    )


def get_plans_to_origin(
    num_plans: int, lims: Tuple[float, float, float, float]
) -> List[PlanarPushingStartAndGoal]:
    # We want the plans to always be the same
    np.random.seed(1)

    x_min, x_max, y_min, y_max = lims

    plans = []
    for _ in range(num_plans):
        x_initial = np.random.uniform(x_min, x_max)
        y_initial = np.random.uniform(y_min, y_max)
        th_initial = np.random.uniform(-np.pi, np.pi)

        pusher_pose = PlanarPose(x_min, y_min, th_initial)

        slider_initial_pose = PlanarPose(x_initial, y_initial, th_initial)
        slider_target_pose = PlanarPose(0, 0, 0)

        plans.append(
            PlanarPushingStartAndGoal(
                slider_initial_pose, slider_target_pose, pusher_pose, pusher_pose
            )
        )

    return plans


def create_plan(
    plan_spec: PlanarPushingStartAndGoal,
    body_to_use: Literal["box", "t_pusher", "sugar_box"] = "sugar_box",
    traj_name: str = "Untitled_traj",
    visualize: bool = False,
    animation_output_dir: str = "",
    animation_lims: Optional[Tuple[float, float, float, float]] = None,
    save_traj: bool = False,
    debug: bool = False,
):
    if body_to_use == "box":
        slider = get_slider_box()
    elif body_to_use == "t_pusher":
        slider = get_tee()
    elif body_to_use == "sugar_box":
        slider = get_sugar_box()

    pusher_radius = 0.035
    dynamics_config = SliderPusherSystemConfig(
        pusher_radius=pusher_radius, slider=slider, friction_coeff_slider_pusher=0.25
    )

    contact_config = ContactConfig(
        cost_type=ContactCostType.OPTIMAL_CONTROL,
        # cost_type=ContactCostType.KEYPOINT_DISPLACEMENTS,
        sq_forces=10.0,
        ang_displacements=1.0,
        lin_displacements=1.0,
        mode_transition_cost=1.0,
        delta_vel_max=0.05,
        delta_theta_max=0.2,
    )

    cost_terms = PlanarCostFunctionTerms(
        obj_avoidance_quad_weight=0.4,
    )

    config = PlanarPlanConfig(
        dynamics_config=dynamics_config,
        cost_terms=cost_terms,
        time_in_contact=7,
        time_non_collision=7,
        num_knot_points_contact=4,
        num_knot_points_non_collision=3,
        avoid_object=True,
        avoidance_cost="quadratic",
        allow_teleportation=False,
        use_band_sparsity=True,
        use_entry_and_exit_subgraphs=True,
        contact_config=contact_config,
    )

    planner = PlanarPushingPlanner(config)

    solver_params = PlanarSolverParams(
        measure_solve_time=True,
        gcs_max_rounded_paths=20,
        print_flows=False,
        print_solver_output=debug,
        save_solver_output=False,
        print_path=True,
        print_cost=True,
        nonlinear_traj_rounding=False,
        assert_result=False,
    )

    planner.config.start_and_goal = plan_spec
    planner.formulate_problem()

    if debug:
        planner.create_graph_diagram(Path("graph.svg"))

    traj = planner.plan_trajectory(solver_params)

    if save_traj:
        filename = f"trajectories/{body_to_use}_pushing_{traj_name}.pkl"
        traj.save(filename)  # type: ignore

    if visualize:
        if animation_output_dir != "":
            traj_name = animation_output_dir + "/" + traj_name

        if debug:
            visualize_planar_pushing_start_and_goal(
                config.slider_geometry,
                config.pusher_radius,
                planner.config.start_and_goal,
                save=True,
                filename=f"{traj_name}_start_and_goal_{body_to_use}",
            )

        ani = visualize_planar_pushing_trajectory(
            traj,  # type: ignore
            save=True,
            filename=f"{traj_name}_{body_to_use}",
            visualize_knot_points=True,
            lims=animation_lims,
        )
        return ani


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--traj",
        help="Which trajectory to plan",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--body",
        help="Which body to plan for",
        type=str,
        default="box",
    )
    parser.add_argument("--demos", help="Generate demos", action="store_true")
    args = parser.parse_args()
    traj_number = args.traj
    make_demos = args.demos

    if make_demos:
        lims = (-0.3, 0.3, -0.3, 0.3)
        animation_lims = (np.array(lims) * 1.3).tolist()
        plans = get_plans_to_origin(9, lims)
        for idx, plan in enumerate(plans):
            create_plan(
                plan,
                debug=False,
                body_to_use=args.body,
                traj_name=f"demo_{idx}",
                visualize=True,
                save_traj=False,
                animation_output_dir="demos",
                animation_lims=animation_lims,
            )

    else:
        plan_spec = get_predefined_plan(traj_number)

        create_plan(
            plan_spec,
            debug=True,
            body_to_use=args.body,
            traj_name=str(traj_number),
            visualize=True,
            save_traj=True,
        )
