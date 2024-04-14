from tqdm import tqdm

from planning_through_contact.experiments.baseline_comparison.direct_trajectory_optimization import (
    direct_trajopt_through_contact,
)
from planning_through_contact.experiments.utils import (
    get_default_experiment_plans,
    get_default_plan_config,
    get_default_solver_params,
)

if __name__ == "__main__":
    traj_idx = None
    # traj = 1

    num_trajs = 50
    seed = 2

    config = get_default_plan_config("sugar_box")
    solver_params = get_default_solver_params(True, clarabel=False)
    solver_params.save_solver_output = True

    plans = get_default_experiment_plans(seed, num_trajs, config)

    if traj_idx is not None:
        res = direct_trajopt_through_contact(
            plans[traj_idx], config, solver_params, output_name=str(traj_idx)
        )
        print(f"result.is_success() = {res}")
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
