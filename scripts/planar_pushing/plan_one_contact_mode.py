import matplotlib.pyplot as plt
from IPython.display import HTML, SVG, display
from pydrake.solvers import (
    ClarabelSolver,
    CommonSolverOption,
    MosekSolver,
    Solve,
    SolverOptions,
)

from planning_through_contact.experiments.utils import (
    get_default_plan_config,
    get_default_solver_params,
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
from planning_through_contact.geometry.planar.trajectory_builder import (
    PlanarTrajectoryBuilder,
)
from planning_through_contact.geometry.rigid_body import RigidBody
from planning_through_contact.planning.planar.planar_plan_config import (
    NonCollisionCost,
    PlanarPlanConfig,
    PlanarPushingStartAndGoal,
    SliderPusherSystemConfig,
)
from planning_through_contact.visualize.analysis import analyze_mode_result
from planning_through_contact.visualize.planar_pushing import (
    visualize_planar_pushing_trajectory,
)

debug = True
slider_type = "box"
# slider_type = "tee"
pusher_radius = 0.035

config = get_default_plan_config(
    slider_type=slider_type,
    pusher_radius=pusher_radius,
    num_knot_points_override=8,
)
config.use_band_sparsity = True
config.use_drake_for_band_sparsity = True

contact_location = PolytopeContactLocation(ContactLocation.FACE, 3)
initial_pose = PlanarPose(0, 0, 0)
final_pose = PlanarPose(0.3, 0.2, 0.4)
config.start_and_goal = PlanarPushingStartAndGoal(initial_pose, final_pose)

mode = FaceContactMode.create_from_plan_spec(contact_location, config)
mode.set_slider_initial_pose(initial_pose)
mode.set_slider_final_pose(final_pose)

mode.formulate_convex_relaxation()

print("Finished formulating convex relaxation")
# solver = ClarabelSolver()
solver = MosekSolver()

solver_options = SolverOptions()
solver_options.SetOption(CommonSolverOption.kPrintToConsole, 1)  # type: ignore

from time import time

start = time()
result = solver.Solve(mode.relaxed_prog, solver_options=solver_options)  # type: ignore
elapsed_time = time() - start
assert result.is_success()
print(f"Cost: {result.get_optimal_cost()}")
print(f"Elapsed time: {elapsed_time}")

vars = mode.variables.eval_result(result)
traj = PlanarPushingTrajectory(config, [vars])

visualize_planar_pushing_trajectory(
    traj, visualize_knot_points=True, save=True, filename="test.mp4"
)
