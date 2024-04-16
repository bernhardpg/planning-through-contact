import argparse

from tqdm import tqdm

from planning_through_contact.experiments.baseline_comparison.direct_trajectory_optimization import (
    SmoothingSchedule,
    direct_trajopt_through_contact,
)
from planning_through_contact.experiments.utils import (
    get_baseline_comparison_configs,
    get_default_experiment_plans,
    get_default_plan_config,
    get_default_solver_params,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        help="Random seed for generating trajectories",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--traj",
        help="(Optional) specify a specific trajectory number to generate, with the given random seed.",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--body",
        help="Which slider body to use.",
        type=str,
        default="box",
    )
    parser.add_argument(
        "--num",
        help="Number of trajectories to generate",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--smooth",
        help="Use smoothing",
        action="store_true",
        default=False,
    )

    args = parser.parse_args()
    seed = args.seed
    traj_number = args.traj
    slider_type = args.body
    num_trajs = args.num
    use_smoothing = args.smooth

    config, solver_params = get_baseline_comparison_configs(slider_type)

    plans = get_default_experiment_plans(seed, num_trajs, config)

    if use_smoothing:
        smoothing = SmoothingSchedule(0.01, 5, "exp")
    else:
        smoothing = None

    if traj_number is not None:
        res = direct_trajopt_through_contact(
            plans[traj_number],
            config,
            solver_params,
            output_name=str(traj_number),
            visualizer="new",
            visualize=True,
            smoothing=smoothing,
        )
        print(f"result.is_success() = {res.is_success()}")
    else:
        found_results = [
            direct_trajopt_through_contact(
                plan, config, solver_params, output_name=str(idx), smoothing=smoothing
            ).is_success()
            for idx, plan in enumerate(tqdm(plans))
        ]
        print(
            f"Found solution in {(sum(found_results) / num_trajs)*100}% of instances."
        )
