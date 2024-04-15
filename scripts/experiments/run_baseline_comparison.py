import argparse
import os

from tqdm import tqdm

from planning_through_contact.experiments.baseline_comparison.direct_trajectory_optimization import (
    direct_trajopt_through_contact,
)
from planning_through_contact.experiments.utils import (
    create_output_folder,
    get_baseline_comparison_configs,
    get_default_experiment_plans,
    get_default_plan_config,
    get_default_solver_params,
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
        default="box",
    )
    parser.add_argument(
        "--num",
        help="Number of trajectories to generate",
        type=int,
        default=100,
    )

    args = parser.parse_args()
    seed = args.seed
    traj_number = args.traj
    slider_type = args.body
    num_trajs = args.num

    config, solver_params = get_baseline_comparison_configs(slider_type)

    plans = get_default_experiment_plans(seed, num_trajs, config)
    output_folder = "baseline_comparison"

    if traj_number is not None:
        plans_to_run = [plans[traj_number]]
    else:
        plans_to_run = plans

    traj_output_folder = create_output_folder(output_folder, slider_type, traj_number)

    for idx, plan in enumerate(plans_to_run):
        output_name = str(idx)
        output_path = f"{traj_output_folder}/{output_name}"
        os.makedirs(output_path, exist_ok=True)

        direct_trajopt_result = direct_trajopt_through_contact(
            plan,
            config,
            solver_params,
            output_name=output_name,
            output_folder=traj_output_folder,
        )
        gcs_solve_data = create_plan(
            plan,
            config,
            solver_params,
            output_name=output_name,
            output_folder=traj_output_folder,
            debug=True,
        )
        with open(f"{output_path}/costs.txt", "w") as f:
            lines = []
            if gcs_solve_data is None:
                lines.append("GCS cost: infeasible")
            else:
                lines.append(f"GCS cost: {gcs_solve_data.feasible_cost}")

            if direct_trajopt_result.is_success():
                lines.append(
                    f"Direct trajopt cost: {direct_trajopt_result.get_optimal_cost()}"
                )
            else:
                lines.append(f"Direct trajopt cost: infeasible")

            for l in lines:
                print(l, file=f)
