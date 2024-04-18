import argparse
import os

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
        "--only_dir",
        help="Only run direct trajopt",
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
    only_direct_trajopt = args.only_dir

    direct_trajopt_config, solver_params = get_baseline_comparison_configs(slider_type)
    gcs_config = direct_trajopt_config

    plans = get_default_experiment_plans(seed, num_trajs, gcs_config)
    output_folder = "baseline_comparison"

    if traj_number is not None:
        plans_to_run = [plans[traj_number]]
    else:
        plans_to_run = plans

    traj_output_folder = create_output_folder(output_folder, slider_type, traj_number)

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
