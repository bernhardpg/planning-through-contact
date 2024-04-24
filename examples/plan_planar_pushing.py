import numpy as np
import pydot

from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.planning.planar.planar_plan_config import (
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
from scripts.planar_pushing.create_plan import get_sugar_box, get_tee

slider = get_sugar_box()
# slider = get_tee()
pusher_radius = 0.035

dynamics_config = SliderPusherSystemConfig(pusher_radius=pusher_radius, slider=slider)

config = PlanarPlanConfig(
    time_non_collision=2.0,
    time_in_contact=2.0,
    num_knot_points_contact=4,
    num_knot_points_non_collision=4,
    avoid_object=True,
    avoidance_cost="socp",
    no_cycles=False,
    dynamics_config=dynamics_config,
    allow_teleportation=False,
    use_redundant_dynamic_constraints=True,
)

planner = PlanarPushingPlanner(config)

solver_params = PlanarSolverParams(
    rounding_steps=10,
    print_flows=False,
    assert_determinants=True,
    print_solver_output=True,
    print_path=True,
    print_cost=True,
    measure_solve_time=True,
    nonlinear_traj_rounding=False,
)

slider_initial_pose = PlanarPose(x=0.55, y=0.0, theta=0.0)
slider_target_pose = PlanarPose(x=0.65, y=0.0, theta=-0.5)
finger_initial_pose = PlanarPose(x=-0.2, y=-0.2, theta=0.0)
finger_target_pose = PlanarPose(x=-0.2, y=-0.2, theta=0.0)
plan = PlanarPushingStartAndGoal(
    slider_initial_pose, slider_target_pose, finger_initial_pose, finger_target_pose
)

planner.set_initial_poses(plan.pusher_initial_pose, plan.slider_initial_pose)
planner.set_target_poses(plan.pusher_target_pose, plan.slider_target_pose)

traj_original = planner.plan_trajectory(solver_params)

visualize_planar_pushing_trajectory(traj_original, visualize_knot_points=True, show=True)  # type: ignore
