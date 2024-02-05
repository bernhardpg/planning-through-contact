from planning_through_contact.experiments.ablation_study.planar_pushing_ablation import (
    do_one_run_get_path,
)
from planning_through_contact.experiments.utils import (
    get_default_plan_config,
    get_default_solver_params,
)
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.planar.planar_pushing_trajectory import (
    PlanarPushingTrajectory,
)
from planning_through_contact.planning.planar.planar_plan_config import (
    PlanarPushingStartAndGoal,
)
from planning_through_contact.visualize.analysis import analyze_plans
from planning_through_contact.visualize.planar_pushing import compare_trajs_vertically
from scripts.planar_pushing.create_plan import get_plans_to_point

make_data = True
if make_data:
    plan_config = get_default_plan_config(slider_type="sugar_box")
    solver_params = get_default_solver_params(debug=True)

    slider_initial_pose = PlanarPose(-0.3, 0.3, 1.7)
    slider_final_pose = PlanarPose(0.0, 0.0, 0.0)
    pusher_initial_pose = PlanarPose(-0.5, 0.0, 0.0)
    pusher_final_pose = PlanarPose(-0.5, 0.0, 0.0)
    start_and_goal = PlanarPushingStartAndGoal(
        slider_initial_pose, slider_final_pose, pusher_initial_pose, pusher_final_pose
    )

    plan_config.contact_config.cost.force_regularization = None
    plan_config.contact_config.cost.lin_velocity_regularization = None
    plan_config.contact_config.cost.ang_velocity_regularization = None
    run_1, path_1 = do_one_run_get_path(plan_config, solver_params, start_and_goal)

    plan_config = get_default_plan_config(slider_type="sugar_box")
    plan_config.contact_config.cost.keypoint_arc_length = None
    run_2, path_2 = do_one_run_get_path(plan_config, solver_params, start_and_goal)

    plan_config = get_default_plan_config(slider_type="sugar_box")
    run_3, path_3 = do_one_run_get_path(plan_config, solver_params, start_and_goal)

    analyze_plans([path_1, path_2, path_3], filename="constraint_violation", legends=["Only arc length", "Only regularization", "Both"])  # type: ignore
    trajs = [path.to_traj() for path in [path_1, path_2, path_3]]
    for idx, traj in enumerate(trajs):
        traj.save(f"traj_{idx}.pkl")


else:
    trajs = [PlanarPushingTrajectory.load(f"traj_{idx}.pkl") for idx in range(3)]
    compare_trajs_vertically(
        trajs,
        filename="qualitative_tightness_figure",
        legends=["Default", "k_f = 0", "k_v = 0", "k_omega = 0"],
    )
