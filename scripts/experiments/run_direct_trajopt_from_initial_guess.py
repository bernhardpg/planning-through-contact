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
        default=None,
    )
    parser.add_argument(
        "--run_dir",
        help="Directory for run.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--traj",
        help="Trajectory in run to use as initial guess. If none is provided, a program will be solved per initial guess in the run_dir",
        type=str,
        default=None,
    )

    parser.add_argument("--debug", help="Debug flag.", action="store_true")

    args = parser.parse_args()
    use_smoothing = args.smooth
    visualize_initial_guess = args.vis_initial
    initial_guess_path = args.guess_path
    run_dir = args.run_dir
    traj = args.traj
    debug = args.debug

    if run_dir is not None:

        initial_guess_paths = []
        if traj is not None:
            traj_folders = [Path(run_dir) / Path(traj)]
        else:
            traj_folders = list(Path(run_dir).iterdir())

        for traj_folder in traj_folders:
            new_initial_guesses = traj_folder.glob("**/*traj*.pkl")
            initial_guess_paths.extend(new_initial_guesses)

        initial_guess_paths = sorted(initial_guess_paths)
    elif initial_guess_path is not None:
        initial_guess_paths = [initial_guess_path]
    else:
        raise RuntimeError("Must provide an initial guess.")

    for initial_guess_path in tqdm(initial_guess_paths):
        traj = PlanarPushingTrajectory.load(initial_guess_path)
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
            use_same_solver_tolerances=False,
            penalize_initial_guess_diff=True,
            save_cost=True,
        )
        print(f"result.is_success() = {res.is_success()}")
