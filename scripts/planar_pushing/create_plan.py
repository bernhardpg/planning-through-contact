import argparse
from pathlib import Path
from typing import Literal

import numpy as np

from planning_through_contact.geometry.collision_geometry.box_2d import Box2d
from planning_through_contact.geometry.collision_geometry.t_pusher_2d import TPusher2d
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.rigid_body import RigidBody
from planning_through_contact.planning.planar.planar_plan_config import (
    PlanarCostFunctionTerms,
    PlanarPlanConfig,
    PlanarSolverParams,
    SliderPusherSystemConfig,
)
from planning_through_contact.planning.planar.planar_pushing_planner import (
    PlanarPushingPlanner,
    PlanarPushingStartAndGoal,
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


def create_plan(
    debug: bool = False,
    body_to_use: Literal["box", "t_pusher", "sugar_box"] = "sugar_box",
    traj_number: int = 1,
    visualize: bool = False,
    save_traj: bool = False,
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

    cost_terms = PlanarCostFunctionTerms(
        sq_forces=10.0,
        ang_displacements=1.0,
        lin_displacements=1.0,
        obj_avoidance_quad_weight=0.4,
        mode_transition_cost=1.0,
    )

    config = PlanarPlanConfig(
        dynamics_config=dynamics_config,
        cost_terms=cost_terms,
        time_in_contact=5,
        time_non_collision=2,
        num_knot_points_contact=3,
        num_knot_points_non_collision=3,
        avoid_object=True,
        avoidance_cost="quadratic",
        allow_teleportation=False,
        use_band_sparsity=True,
        minimize_sq_forces=True,
        use_entry_and_exit_subgraphs=True,
        penalize_mode_transitions=False,
        minimize_keypoint_displacement=True,
    )

    planner = PlanarPushingPlanner(config)

    solver_params = PlanarSolverParams(
        measure_solve_time=True,
        gcs_max_rounded_paths=20,
        print_flows=False,
        print_solver_output=True,
        save_solver_output=False,
        print_path=True,
        print_cost=True,
        nonlinear_traj_rounding=False,
        assert_result=False,
    )

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
        slider_initial_pose = PlanarPose(x=0.45, y=-0.1, theta=np.pi/2)
        slider_target_pose = PlanarPose(x=0.45, y=0.1, theta=np.pi/2)
        pusher_initial_pose = PlanarPose(x=-0.2, y=0, theta=0.0)
        pusher_target_pose = PlanarPose(x=-0.2, y=0, theta=0.0)
    elif traj_number == 14: # Fails
        slider_initial_pose = PlanarPose(x=0.45, y=-0.1, theta=0.9)
        slider_target_pose = PlanarPose(x=0.35, y=0.1, theta=np.pi/2)
        pusher_initial_pose = PlanarPose(x=-0.2, y=0, theta=0.0)
        pusher_target_pose = PlanarPose(x=-0.2, y=0, theta=0.0)
    elif traj_number == 15: # Fails
        slider_initial_pose = PlanarPose(x=0.50, y=-0.1, theta=np.pi/2)
        slider_target_pose = PlanarPose(x=0.45, y=0.1, theta=1.1)
        pusher_initial_pose = PlanarPose(x=-0.2, y=0, theta=0.0)
        pusher_target_pose = PlanarPose(x=-0.2, y=0, theta=0.0)
    elif traj_number == 16: # Fails
        slider_initial_pose = PlanarPose(x=0.45, y=0.1, theta=1.1)
        slider_target_pose = PlanarPose(x=0.50, y=-0.1, theta=np.pi/2)
        pusher_initial_pose = PlanarPose(x=0.2, y=0, theta=0.0)
        pusher_target_pose = PlanarPose(x=0.2, y=0, theta=0.0)
    elif traj_number == 17:
        slider_initial_pose = PlanarPose(x=0.45, y=0.1, theta=1.2)
        slider_target_pose = PlanarPose(x=0.60, y=0.0, theta=np.pi/2)
        pusher_initial_pose = PlanarPose(x=0.2, y=0, theta=0.0)
        pusher_target_pose = PlanarPose(x=0.2, y=0, theta=0.0)
    elif traj_number == 18:
        slider_target_pose = PlanarPose(x=0.45, y=-0.1, theta=np.pi / 2)
        pusher_initial_pose = PlanarPose(x=-0.2, y=-0.2, theta=0.0)
        pusher_target_pose = PlanarPose(x=-0.2, y=-0.2, theta=0.0)
    else:
        raise NotImplementedError()

    planner.set_initial_poses(pusher_initial_pose, slider_initial_pose)
    planner.set_target_poses(pusher_target_pose, slider_target_pose)
    planner.formulate_problem()

    if debug:
        planner.create_graph_diagram(Path("graph.svg"))

    traj = planner.plan_trajectory(solver_params)

    if save_traj:
        traj_name = f"trajectories/{body_to_use}_pushing_{traj_number}.pkl"
        traj.save(traj_name)  # type: ignore

    if visualize:
        plan = PlanarPushingStartAndGoal(
            slider_initial_pose,
            slider_target_pose,
            pusher_initial_pose,
            pusher_target_pose,
        )
        visualize_planar_pushing_start_and_goal(
            config.slider_geometry,
            config.pusher_radius,
            plan,
            save=True,
            filename=f"trajectory_{traj_number}_start_and_goal_{body_to_use}",
        )

        ani = visualize_planar_pushing_trajectory(
            traj,  # type: ignore
            save=True,
            filename=f"trajectory_{traj_number}_{body_to_use}",
            visualize_knot_points=True,
            show=True,
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
    args = parser.parse_args()
    traj_number = args.traj
    create_plan(
        debug=True,
        body_to_use=args.body,
        traj_number=traj_number,
        visualize=False,
        save_traj=True,
    )
