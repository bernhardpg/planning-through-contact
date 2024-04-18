import argparse
from pathlib import Path

from tqdm import tqdm

from planning_through_contact.experiments.baseline_comparison.direct_trajectory_optimization import (
    SmoothingSchedule,
    direct_trajopt_through_contact,
)
from planning_through_contact.experiments.utils import (
    get_baseline_comparison_configs,
    get_default_baseline_solver_params,
    get_default_experiment_plans,
)
from planning_through_contact.geometry.planar.planar_pushing_trajectory import (
    PlanarPushingTrajectory,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smooth",
        help="Use smoothing",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--vis_initial",
        help="Visualize initial guess",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--guess_path",
        help="Provide a .pkl file as the initial guess",
        type=str,
        required=True,
    )

    debug = True

    args = parser.parse_args()
    use_smoothing = args.smooth
    visualize_initial_guess = args.vis_initial
    initial_guess_path = Path(args.guess_path)

    traj = PlanarPushingTrajectory.load(str(initial_guess_path))
    config = traj.config

    # config.contact_config.cost.force_regularization = None
    # config.contact_config.cost.keypoint_arc_length = None
    # config.contact_config.cost.keypoint_velocity_regularization = None
    # config.contact_config.cost.time = None
    #
    # config.non_collision_cost.pusher_velocity_regularization = None
    # config.non_collision_cost.pusher_arc_length = None
    # config.non_collision_cost.distance_to_object_socp = None
    # config.non_collision_cost.time = None

    solver_params = get_default_baseline_solver_params()
    plan = config.start_and_goal

    assert plan is not None

    if use_smoothing:
        smoothing = SmoothingSchedule(0.01, 5, "exp", "decreasing")
    else:
        smoothing = None

    res = direct_trajopt_through_contact(
        plan,
        config,
        solver_params,
        output_folder=str(initial_guess_path.parent),
        output_name="direct_from_initial_guess",
        visualizer="new",
        visualize=True,
        smoothing=smoothing,
        visualize_initial_guess=visualize_initial_guess,
        initial_guess=traj,
        debug=debug,
        use_same_solver_tolerances=True,
        penalize_initial_guess_diff=True,
    )
    print(f"result.is_success() = {res.is_success()}")
