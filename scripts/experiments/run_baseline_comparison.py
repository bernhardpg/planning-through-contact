import argparse

from tqdm import tqdm

from planning_through_contact.experiments.baseline_comparison.direct_trajectory_optimization import (
    direct_trajopt_through_contact,
)
from planning_through_contact.experiments.utils import (
    create_output_folder,
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
    parser.add_argument(
        "--debug",
        help="Debug mode. Will print additional information, including solver output.",
        action="store_true",
    )

    args = parser.parse_args()
    seed = args.seed
    traj_number = args.traj
    debug = args.debug
    slider_type = args.body
    num_trajs = args.num

    config = get_default_plan_config(slider_type)
    solver_params = get_default_solver_params(True, clarabel=False)
    solver_params.save_solver_output = True

    plans = get_default_experiment_plans(seed, num_trajs, config)
    output_folder = "baseline_comparison"

    if traj_number is not None:
        traj_output_folder = create_output_folder(
            output_folder, slider_type, traj_number
        )

        plan = plans[traj_number]
        output_name = str(traj_number)
        res = direct_trajopt_through_contact(
            plan,
            config,
            solver_params,
            output_name=output_name,
            output_folder=traj_output_folder,
        )
        create_plan(
            plan,
            config,
            solver_params,
            output_name=output_name,
            output_folder=traj_output_folder,
        )

    else:
        found_results = [
            direct_trajopt_through_contact(
                plan, config, solver_params, output_name=str(idx)
            )
            for idx, plan in enumerate(tqdm(plans))
        ]
        print(
            f"Found solution in {(sum(found_results) / num_trajs)*100}% of instances."
        )
