import numpy as np
from pydrake.solvers import MosekSolver

from planning_through_contact.experiments.utils import (
    get_default_contact_cost,
    get_default_plan_config,
)
from planning_through_contact.geometry.collision_geometry.box_2d import Box2d
from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    ContactLocation,
    PolytopeContactLocation,
)
from planning_through_contact.geometry.planar.face_contact import FaceContactMode
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.planar.planar_pushing_trajectory import (
    PlanarPushingTrajectory,
)
from planning_through_contact.geometry.rigid_body import RigidBody
from planning_through_contact.planning.planar.planar_plan_config import (
    ContactCost,
    PlanarPushingStartAndGoal,
)
from planning_through_contact.visualize.analysis import (
    analyze_mode_result,
    analyze_plan,
    analyze_plans,
    get_constraint_violation_for_face_mode,
    plot_constraint_violation_for_trajs,
)
from planning_through_contact.visualize.colors import (
    AQUAMARINE3,
    AQUAMARINE4,
    BROWN2,
    BURLYWOOD3,
    DARKORCHID3,
    DODGERBLUE2,
)
from planning_through_contact.visualize.planar_pushing import (
    make_traj_figure,
    visualize_planar_pushing_trajectory,
)


def plan(contact_location, plan_cfg, initial_pose, final_pose):
    mode = FaceContactMode.create_from_plan_spec(contact_location, plan_cfg)
    mode.set_slider_initial_pose(initial_pose)
    mode.set_slider_final_pose(final_pose)
    mode.add_so2_cut_from_boundary_conds(add_as_independent=True)

    mode.formulate_convex_relaxation()
    solver = MosekSolver()
    result = solver.Solve(mode.relaxed_prog)  # type: ignore
    assert result.is_success()

    vars = mode.variables.eval_result(result)
    traj = PlanarPushingTrajectory(plan_cfg, [vars])

    return traj, mode, result


colors = [
    BROWN2.diffuse(),
    AQUAMARINE4.diffuse(),
    DODGERBLUE2.diffuse(),
]


box_geometry = Box2d(width=0.3, height=0.3)
mass = 0.3
box = RigidBody("box", box_geometry, mass)

config = get_default_plan_config()
config.num_knot_points_contact = 4
config.use_band_sparsity = True
config.dynamics_config.pusher_radius = 0.05
config.dynamics_config.slider = box
config.dynamics_config.force_scale = 1.0
contact_location = PolytopeContactLocation(ContactLocation.FACE, 3)
initial_pose = PlanarPose(0, 0, 0)
final_pose = PlanarPose(0.3, 0.2, 1.2)
config.start_and_goal = PlanarPushingStartAndGoal(initial_pose, final_pose)

# No ang vel reg
config.contact_config.cost = ContactCost(
    keypoint_arc_length=None,
    ang_velocity_regularization=None,
    lin_velocity_regularization=1.0,
)

traj, mode, result = plan(contact_location, config, initial_pose, final_pose)
cost = result.get_optimal_cost()
mode_vars = mode.get_variable_solutions(result)
print(f"Force norms: {[np.linalg.norm(f) for f in mode_vars.f_c_Bs]}")
print(f"Normal components: {mode_vars.normal_forces}")
print(f"Friction components: {mode_vars.friction_forces}")
print(f"Cost: {cost}")

name = "no_ang_vel_reg"
ani = visualize_planar_pushing_trajectory(
    traj, visualize_knot_points=True, save=True, filename=name
)
make_traj_figure(traj, filename=name, plot_lims=None, start_end_legend=False)
analyze_mode_result(mode, traj, result, filename=name)

constraint_violations_1 = get_constraint_violation_for_face_mode(
    mode,
    result,
    compute_mean=True,
    keys_to_merge=[("translational_dynamics", "translational_dynamics_red")],
)

# ang vel reg
config.contact_config.cost = ContactCost(
    keypoint_arc_length=None,
    ang_velocity_regularization=1.0,
    lin_velocity_regularization=1.0,
)

traj, mode, result = plan(contact_location, config, initial_pose, final_pose)
cost = result.get_optimal_cost()
mode_vars = mode.get_variable_solutions(result)
print(f"Force norms: {[np.linalg.norm(f) for f in mode_vars.f_c_Bs]}")
print(f"Normal components: {mode_vars.normal_forces}")
print(f"Friction components: {mode_vars.friction_forces}")
print(f"Cost: {cost}")

name = "ang_vel_reg"
ani = visualize_planar_pushing_trajectory(
    traj, visualize_knot_points=True, save=True, filename=name
)
make_traj_figure(traj, filename=name, plot_lims=None, start_end_legend=False)
analyze_mode_result(mode, traj, result, filename=name)

constraint_violations_2 = get_constraint_violation_for_face_mode(
    mode,
    result,
    compute_mean=True,
    keys_to_merge=[("translational_dynamics", "translational_dynamics_red")],
)

# force reg
config.contact_config.cost = ContactCost(
    ang_velocity_regularization=1.0,
    force_regularization=1.0,
    lin_velocity_regularization=1.0,
)

traj, mode, result = plan(contact_location, config, initial_pose, final_pose)
cost = result.get_optimal_cost()
mode_vars = mode.get_variable_solutions(result)
print(f"Force norms: {[np.linalg.norm(f) for f in mode_vars.f_c_Bs]}")
print(f"Normal components: {mode_vars.normal_forces}")
print(f"Friction components: {mode_vars.friction_forces}")
print(f"Cost: {cost}")

name = "force_reg"
ani = visualize_planar_pushing_trajectory(
    traj, visualize_knot_points=True, save=True, filename=name
)
make_traj_figure(traj, filename=name, plot_lims=None, start_end_legend=True)
analyze_mode_result(mode, traj, result, filename=name)

constraint_violations_3 = get_constraint_violation_for_face_mode(
    mode,
    result,
    compute_mean=True,
    keys_to_merge=[("translational_dynamics", "translational_dynamics_red")],
)

import matplotlib as mpl

# Enable LaTeX in Matplotlib
mpl.rcParams["text.usetex"] = True
bar_titles = [
    "$(14)$",
    "$(11)$ (Rotational part)",
    "$(11)$ (Translational part)",
]
plot_constraint_violation_for_trajs(
    [constraint_violations_1, constraint_violations_2, constraint_violations_3],
    bar_titles=bar_titles,
    filename="comparing_violations",
    legends=[
        "No regularization",
        "$k_\omega = 1, \, k_f = 0$",
        "$k_\omega = 1, k_f=1$",
    ],
    min_y_ax=1e-4,
    colors=colors,
)
