import numpy as np
import pydot
from IPython.display import HTML, SVG, display

from planning_through_contact.geometry.planar.planar_pose import PlanarPose
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
from scripts.planar_pushing.create_plan import get_sugar_box, get_tee

# slider = get_sugar_box()
slider = get_tee()
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
    num_knot_points_contact=5,
    num_knot_points_non_collision=3,
    avoid_object=True,
    avoidance_cost="quadratic",
    allow_teleportation=False,
    use_band_sparsity=True,
    minimize_sq_forces=True,
    use_entry_and_exit_subgraphs=True,
    penalize_mode_transitions=False,
    # no_cycles=True,
)

planner = PlanarPushingPlanner(config)

solver_params = PlanarSolverParams(
    gcs_max_rounded_paths=20,
    print_flows=False,
    print_solver_output=True,
    save_solver_output=False,
    print_path=True,
    print_cost=True,
    nonlinear_traj_rounding=False,
    assert_result=False,
)


# Traj 1
# slider_initial_pose = PlanarPose(x=0.0, y=0.0, theta=0.0)
# slider_target_pose = PlanarPose(x=0.3, y=-0.15, theta=-0.5)
# finger_initial_pose = PlanarPose(x=0.0, y=0.2, theta=0.0)
# finger_target_pose = PlanarPose(x=0.0, y=0.2, theta=0.0)

# Traj 2
slider_initial_pose = PlanarPose(x=0.0, y=0.0, theta=0.0)
slider_target_pose = PlanarPose(x=-0.3, y=-0.15, theta=-0.5)
finger_initial_pose = PlanarPose(x=0.0, y=0.2, theta=0.0)
finger_target_pose = PlanarPose(x=0.0, y=0.2, theta=0.0)

# Traj 3
# slider_initial_pose = PlanarPose(x=0.5, y=0.0, theta=1.0)
# slider_target_pose = PlanarPose(x=0.6, y=-0.15, theta=-0.5)
# finger_initial_pose = PlanarPose(x=0.0, y=0.2, theta=0.0)
# finger_target_pose = PlanarPose(x=0.0, y=0.2, theta=0.0)

# traj 4
# slider_initial_pose = PlanarPose(x=0.7, y=0.2, theta=0.3)
# slider_target_pose = PlanarPose(x=0.55, y=-0.15, theta=1.2)
# finger_initial_pose = PlanarPose(x=0.0, y=0.2, theta=0.0)
# finger_target_pose = PlanarPose(x=0.0, y=0.2, theta=0.0)
plan = PlanarPushingStartAndGoal(
    slider_initial_pose, slider_target_pose, finger_initial_pose, finger_target_pose
)

planner.set_initial_poses(plan.pusher_initial_pose, plan.slider_initial_pose)
planner.set_target_poses(plan.pusher_target_pose, plan.slider_target_pose)
planner.formulate_problem()

traj = planner.plan_trajectory(solver_params)
visualize_planar_pushing_trajectory(
    traj, visualize_knot_points=True, save=True, filename="generated_trajectory"  # type: ignore
)
