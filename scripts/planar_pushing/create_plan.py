import argparse
from pathlib import Path
from typing import Literal

import numpy as np

from planning_through_contact.geometry.collision_geometry.box_2d import Box2d
from planning_through_contact.geometry.collision_geometry.t_pusher_2d import TPusher2d
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.rigid_body import RigidBody
from planning_through_contact.planning.planar.planar_plan_config import (
    PlanarPlanConfig,
    PlanarSolverParams,
    SliderPusherSystemConfig,
)
from planning_through_contact.planning.planar.planar_pushing_planner import (
    PlanarPushingPlanner,
)
from planning_through_contact.visualize.planar import (
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
        body = get_slider_box()
    elif body_to_use == "t_pusher":
        body = get_tee()
    elif body_to_use == "sugar_box":
        body = get_sugar_box()

    dynamics_config = SliderPusherSystemConfig(pusher_radius=0.035, slider=body)

    config = PlanarPlanConfig(
        time_non_collision=2.0,
        time_in_contact=2.0,
        num_knot_points_contact=4,
        num_knot_points_non_collision=4,
        avoid_object=True,
        avoidance_cost="quadratic",
        no_cycles=True,
        dynamics_config=dynamics_config,
        allow_teleportation=False,
    )

    planner = PlanarPushingPlanner(config)

    if traj_number == 1:  # loose
        slider_initial_pose = PlanarPose(x=0.55, y=0.0, theta=0.0)
        slider_target_pose = PlanarPose(x=0.65, y=0.0, theta=-0.5)
        finger_initial_pose = PlanarPose(x=-0.2, y=-0.2, theta=0.0)
        finger_target_pose = PlanarPose(x=-0.2, y=-0.2, theta=0.0)
    elif traj_number == 2:  # works
        slider_initial_pose = PlanarPose(x=0.60, y=0.1, theta=-0.2)
        slider_target_pose = PlanarPose(x=0.70, y=-0.2, theta=0.5)
        finger_initial_pose = PlanarPose(x=-0.2, y=-0.2, theta=0.0)
        finger_target_pose = PlanarPose(x=-0.2, y=-0.2, theta=0.0)
    elif traj_number == 3:  # crazy movement
        slider_initial_pose = PlanarPose(x=0.60, y=0.1, theta=-0.2)
        slider_target_pose = PlanarPose(x=0.70, y=-0.2, theta=0.5)
        finger_initial_pose = PlanarPose(x=-0.2, y=0.0, theta=0.0)
        finger_target_pose = PlanarPose(x=-0.2, y=0.0, theta=0.0)
    elif traj_number == 4:
        slider_initial_pose = PlanarPose(x=0.60, y=0.1, theta=np.pi / 2 - 0.2)
        slider_target_pose = PlanarPose(x=0.75, y=-0.2, theta=np.pi / 2 + 0.4)
        finger_initial_pose = PlanarPose(x=-0.2, y=0.0, theta=0.0)
        finger_target_pose = PlanarPose(x=0.0, y=0.2, theta=0.0)
    elif traj_number == 5:  # loose
        slider_initial_pose = PlanarPose(x=0.60, y=0.1, theta=np.pi / 2)
        slider_target_pose = PlanarPose(x=0.70, y=-0.05, theta=np.pi / 2 + 0.4)
        finger_initial_pose = PlanarPose(x=-0.2, y=0.0, theta=0.0)
        finger_target_pose = PlanarPose(x=0.0, y=0.2, theta=0.0)
    elif traj_number == 6:  # t pusher
        slider_initial_pose = PlanarPose(x=0.60, y=0.0, theta=np.pi / 2)
        slider_target_pose = PlanarPose(x=0.65, y=-0.1, theta=np.pi / 2 + 0.3)
        finger_initial_pose = PlanarPose(x=-0.2, y=0.0, theta=0.0)
        finger_target_pose = PlanarPose(x=0.0, y=0.2, theta=0.0)
    else:
        raise NotImplementedError()

    planner.set_initial_poses(finger_initial_pose, slider_initial_pose)
    planner.set_target_poses(finger_target_pose, slider_target_pose)

    if debug:
        planner.create_graph_diagram(Path("graph.svg"))

    solver_params = PlanarSolverParams(
        gcs_max_rounded_paths=10,
        print_flows=True,
        nonlinear_traj_rounding=False,
        assert_determinants=False,
        print_solver_output=True,
        print_path=True,
    )
    traj = planner.plan_trajectory(solver_params)

    if save_traj:
        traj_name = f"trajectories/{body_to_use}_pushing_{traj_number}.pkl"
        traj.save(traj_name)

    if visualize:
        ani = visualize_planar_pushing_trajectory(
            traj, save=True, filename="test", visualize_knot_points=True
        )
        return ani
        # visualize_planar_pushing_trajectory_legacy(
        #     traj.to_old_format(),
        #     body.geometry,
        #     config.pusher_radius,
        #     visualize_robot_base=True,
        # )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--traj",
        help="Which trajectory to plan",
        type=int,
        default=1,
    )
    args = parser.parse_args()
    traj_number = args.traj
    create_plan(debug=True, body_to_use="box", traj_number=traj_number, visualize=True)
