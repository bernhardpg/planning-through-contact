import matplotlib.pyplot as plt
from IPython.display import HTML, SVG, display
from pydrake.solvers import CommonSolverOption, MosekSolver, Solve, SolverOptions

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
from planning_through_contact.geometry.planar.trajectory_builder import (
    PlanarTrajectoryBuilder,
)
from planning_through_contact.geometry.rigid_body import RigidBody
from planning_through_contact.planning.planar.planar_plan_config import (
    PlanarCostFunctionTerms,
    PlanarPlanConfig,
    SliderPusherSystemConfig,
)
from planning_through_contact.visualize.analysis import analyze_mode_result
from planning_through_contact.visualize.planar_pushing import (
    visualize_planar_pushing_trajectory,
)

box_geometry = Box2d(width=0.3, height=0.3)
mass = 0.3
box = RigidBody("box", box_geometry, mass)
cfg = SliderPusherSystemConfig(
    slider=box,
    pusher_radius=0.05,
    friction_coeff_slider_pusher=0.5,
    friction_coeff_table_slider=0.5,
    integration_constant=0.7,
)
cost_terms = PlanarCostFunctionTerms(cost_param_forces=1.0, cost_param_ang_vels=1.0)
plan_cfg = PlanarPlanConfig(
    dynamics_config=cfg,
    num_knot_points_contact=40,
    cost_terms=cost_terms,
    use_approx_exponential_map=True,
    use_band_sparsity=False,
)


contact_location = PolytopeContactLocation(ContactLocation.FACE, 3)
initial_pose = PlanarPose(0, 0, 0)
final_pose = PlanarPose(0.3, 0.1, 0.4)

mode = FaceContactMode.create_from_plan_spec(contact_location, plan_cfg)
mode.set_slider_initial_pose(initial_pose)
mode.set_slider_final_pose(final_pose)

mode.formulate_convex_relaxation()

print("Finished formulating convex relaxation")
solver = MosekSolver()

solver_options = SolverOptions()
solver_options.SetOption(CommonSolverOption.kPrintToConsole, 1)  # type: ignore

from time import time

start = time()
result = Solve(mode.relaxed_prog, solver_options=solver_options)  # type: ignore
elapsed_time = time() - start
assert result.is_success()
print(f"Cost: {result.get_optimal_cost()}")
print(f"Elapsed time: {elapsed_time}")

vars = mode.variables.eval_result(result)
traj = PlanarPushingTrajectory(plan_cfg, [vars])

visualize_planar_pushing_trajectory(
    traj, visualize_knot_points=True, save=True, filename="test.mp4"
)
