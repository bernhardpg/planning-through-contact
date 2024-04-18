import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from planning_through_contact.experiments.baseline_comparison.direct_trajectory_optimization import (
    SmoothingSchedule,
    direct_trajopt_through_contact,
)
from planning_through_contact.experiments.utils import (
    create_output_folder,
    get_baseline_comparison_configs,
    get_default_experiment_plans,
    get_default_plan_config,
)
from planning_through_contact.planning.planar.utils import create_plan


@dataclass
class ComparisonRunData:
    num_total_runs: int
    num_gcs_success: int
    num_direct_trajopt_success: int

    @property
    def percentage_gcs_success(self) -> float:
        return (self.num_gcs_success / self.num_total_runs) * 100

    @property
    def percentage_direct_success(self) -> float:
        return (self.num_direct_trajopt_success / self.num_total_runs) * 100

    def print_stats(self) -> None:
        print(f"Total number of runs: {self.num_total_runs}")
        print(
            f"Total number of GCS successes: {self.num_gcs_success} ({self.percentage_gcs_success:.2f} %)"
        )
        print(
            f"Total number of direct trajopt successes: {self.num_direct_trajopt_success} ({self.percentage_direct_success:.2f} %)"
        )


def _count_successes(run_dir: str) -> ComparisonRunData:
    run_dir_path = Path(run_dir)

    num_gcs_success = 0
    num_direct_trajopt_success = 0
    num_total_runs = 0
    for traj_folder in run_dir_path.iterdir():
        cost_files = list(traj_folder.glob("costs.txt"))
        if len(cost_files) == 0:
            continue  # if there is no cost file then this trajectory was not completed, and we skip this folder
        else:
            num_total_runs += 1
            cost_file = cost_files[0]
            with open(cost_file) as f:
                lines = list(f)
                gcs_result = lines[0]
                direct_result = lines[1]
                if "infeasible" in gcs_result or "not_run" in gcs_result:
                    ...  # nothing to do
                else:
                    num_gcs_success += 1

                if "infeasible" in direct_result or "not_run" in direct_result:
                    ...  # nothing to do
                else:
                    num_direct_trajopt_success += 1

    return ComparisonRunData(
        num_total_runs, num_gcs_success, num_direct_trajopt_success
    )


def main() -> None:
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
        default="sugar_box",
    )
    parser.add_argument(
        "--num",
        help="Number of trajectories to generate",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--smooth",
        help="Use smoothing",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--debug",
        help="Debug mode. Will print additional information",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--only_direct",
        help="Only run direct trajopt",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--stats",
        help="Print statistics for an existing baseline comparison run.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--run_dir",
        help="Directory to print statistics for",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--vis_initial",
        help="Visualize initial guess",
        action="store_true",
        default=False,
    )

    args = parser.parse_args()
    seed = args.seed
    traj_number = args.traj
    debug = args.debug
    slider_type = args.body
    num_trajs = args.num
    use_smoothing = args.smooth
    only_direct_trajopt = args.only_direct
    print_stats = args.stats
    run_dir = args.run_dir
    visualize_initial_guess = args.vis_initial

    # This plan works for the Tee
    # slider_initial_pose = PlanarPose(-0.1, 0, 0.3)
    # slider_target_pose = PlanarPose(0, 0, 0)
    # pusher_initial_pose = PlanarPose(-0.5, 0.04, 0)
    # pusher_target_pose = PlanarPose(-0.5, 0.04, 0)

    if print_stats:
        if run_dir is None:
            raise RuntimeError("Must provide a directory to print statistics from.")
        run_data = _count_successes(run_dir)
        run_data.print_stats()
        return

    direct_trajopt_config, solver_params = get_baseline_comparison_configs(slider_type)
    gcs_config = direct_trajopt_config

    plans = get_default_experiment_plans(seed, num_trajs, gcs_config)
    output_folder = "baseline_comparison"

    if traj_number is not None:
        plans_to_run = [plans[traj_number]]
    else:
        plans_to_run = plans

    traj_output_folder = create_output_folder(output_folder, slider_type, traj_number)

    with open(f"{traj_output_folder}/seed.txt", "w") as f:
        print(f"seed: {seed}", file=f)

    for idx, plan in enumerate(tqdm(plans_to_run)):
        output_name = str(idx)
        output_path = f"{traj_output_folder}/{output_name}"
        os.makedirs(output_path, exist_ok=True)

        if use_smoothing:
            smoothing = SmoothingSchedule(0.01, 5, "exp")
        else:
            smoothing = None

        direct_trajopt_result = direct_trajopt_through_contact(
            plan,
            direct_trajopt_config,
            solver_params,
            output_name=output_name,
            output_folder=traj_output_folder,
            smoothing=smoothing,
            debug=debug,
            visualize_initial_guess=visualize_initial_guess,
        )
        if only_direct_trajopt:
            gcs_solve_data = None
        else:
            gcs_solve_data = create_plan(
                plan,
                gcs_config,
                solver_params,
                output_name=output_name,
                output_folder=traj_output_folder,
                debug=debug,
            )
        with open(f"{output_path}/costs.txt", "w") as f:
            lines = []
            if only_direct_trajopt:
                gcs_cost = "not_run"
            elif gcs_solve_data is None:
                gcs_cost = "infeasible"
            else:
                gcs_cost = gcs_solve_data.feasible_cost

            lines.append(f"gcs_cost: {gcs_cost}")

            if direct_trajopt_result.is_success():
                dir_trajopt_cost = direct_trajopt_result.get_optimal_cost()
            else:
                dir_trajopt_cost = "infeasible"

            lines.append(f"dir_trajopt_cost: {dir_trajopt_cost}")

            for l in lines:
                print(l, file=f)


if __name__ == "__main__":
    main()
