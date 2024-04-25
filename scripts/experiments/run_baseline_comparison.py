import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

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
)
from planning_through_contact.geometry.collision_geometry.box_2d import Box2d
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.planar.planar_pushing_trajectory import (
    PlanarPushingTrajectory,
    SimplePlanarPushingTrajectory,
)
from planning_through_contact.planning.planar.planar_plan_config import (
    PlanarPushingStartAndGoal,
)
from planning_through_contact.planning.planar.utils import create_plan
from planning_through_contact.visualize.planar_pushing import (
    visualize_initial_conditions,
)


@dataclass
class ComparisonRunData:
    run_dir: Path
    num_total_runs: int
    gcs_successes: List[str]
    gcs_failures: List[str]
    direct_trajopt_successes: List[str]
    direct_trajopt_failures: List[str]

    def __post_init__(self) -> None:
        self.gcs_successes = sorted(self.gcs_successes)
        self.direct_trajopt_successes = sorted(self.direct_trajopt_successes)

    @property
    def num_gcs_success(self) -> int:
        return len(self.gcs_successes)

    @property
    def num_direct_trajopt_success(self) -> int:
        return len(self.direct_trajopt_successes)

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

    def print_successes(self) -> None:
        print("Succesfull gcs: " + ", ".join(self.gcs_successes))
        print("Succesfull direct trajopt: " + ", ".join(self.direct_trajopt_successes))

    def load_traj(
        self, name: str
    ) -> PlanarPushingTrajectory | SimplePlanarPushingTrajectory | None:
        traj_folder = self.run_dir / name
        trajs = list(traj_folder.glob("**/traj_rounded.pkl"))
        if len(trajs) == 1:
            traj_path = trajs[0]
            traj = PlanarPushingTrajectory.load(str(traj_path))
        else:
            trajs = list(traj_folder.glob("**/direct_traj.pkl"))
            if len(trajs) == 1:
                traj_path = trajs[0]
                traj = SimplePlanarPushingTrajectory.load(str(traj_path))
            else:
                return None

        return traj

    def load_initial_conditions(self, name: str) -> PlanarPushingStartAndGoal | None:
        traj = self.load_traj(name)
        if traj is not None:
            assert traj.config.start_and_goal is not None
            return traj.config.start_and_goal
        else:
            return None

    @classmethod
    def load_from_run(cls, run_dir: str) -> "ComparisonRunData":
        run_dir_path = Path(run_dir)

        gcs_successes = []
        gcs_failures = []
        direct_trajopt_successes = []
        direct_trajopt_failures = []

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
                    direct_trajopt_result = lines[1]
                    if "infeasible" in gcs_result:
                        gcs_failures.append(traj_folder.name)
                    elif "not_run" in gcs_result:
                        ...  # nothing to do
                    else:
                        gcs_successes.append(traj_folder.name)

                    if "infeasible" in direct_trajopt_result:
                        direct_trajopt_failures.append(traj_folder.name)
                    elif "not_run" in direct_trajopt_result:
                        ...  # nothing to do
                    else:
                        direct_trajopt_successes.append(traj_folder.name)

        return cls(
            run_dir_path,
            num_total_runs,
            gcs_successes,
            gcs_failures,
            direct_trajopt_successes,
            direct_trajopt_failures,
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
        help="Use smoothing with a had constraint on the relaxation level.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--soft_smooth",
        help="Use smoothing with a penalty on the violation rather than a constraint.",
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
    parser.add_argument(
        "--vis_failures",
        help="Gather all the plans in a run and visualize all the initial configurations that failed.",
        action="store_true",
    )
    parser.add_argument(
        "--vis_success",
        help="Gather all the plans in a run and visualize all the initial configurations that failed.",
        action="store_true",
    )

    args = parser.parse_args()
    seed = args.seed
    traj_number = args.traj
    debug = args.debug
    slider_type = args.body
    num_trajs = args.num
    use_hard_smoothing = args.smooth
    use_soft_smoothing = args.soft_smooth
    only_direct_trajopt = args.only_direct
    print_stats = args.stats
    run_dir = args.run_dir
    visualize_initial_guess = args.vis_initial
    visualize_failures = args.vis_failures
    visualize_success = args.vis_success

    if print_stats or visualize_success or visualize_failures:
        if run_dir is None:
            raise RuntimeError("Must provide a directory to print statistics from.")
        run_data = ComparisonRunData.load_from_run(run_dir)

        if print_stats:
            run_data.print_stats()
            run_data.print_successes()

        if visualize_success or visualize_failures:
            if visualize_failures:
                trajs_to_vis = run_data.direct_trajopt_failures
                filename = "direct_trajopt_failures"
            else:  # visualize_success
                trajs_to_vis = run_data.direct_trajopt_successes
                filename = "direct_trajopt_successes"

            initial_conds = [
                run_data.load_initial_conditions(traj) for traj in trajs_to_vis
            ]
            initial_conds = [c for c in initial_conds if c is not None]
            # use the config from the first traj
            config = run_data.load_traj(trajs_to_vis[0]).config
            visualize_initial_conditions(
                initial_conds,
                config,
                filename=f"{run_dir}/{filename}",
                plot_orientation_arrow=(
                    True if type(config.slider_geometry) == Box2d else False
                ),
            )

        return

    direct_trajopt_config, solver_params = get_baseline_comparison_configs(
        slider_type, use_velocity_limits=True
    )
    gcs_config = direct_trajopt_config

    plans = get_default_experiment_plans(seed, num_trajs, gcs_config)
    output_folder = "baseline_comparison"

    if traj_number is not None:
        plans_to_run = [plans[traj_number]]
    else:
        plans_to_run = plans

    traj_output_folder = create_output_folder(output_folder, slider_type, traj_number)

    with open(f"{traj_output_folder}/run_specs.txt", "w") as f:
        print(f"seed: {seed}", file=f)
        print(f"use_hard_smoothing: {use_hard_smoothing}", file=f)
        print(f"use_soft_smoothing: {use_soft_smoothing}", file=f)

    for idx, plan in enumerate(tqdm(plans_to_run)):
        output_name = str(idx)
        output_path = f"{traj_output_folder}/{output_name}"
        os.makedirs(output_path, exist_ok=True)

        if use_hard_smoothing:
            smoothing = SmoothingSchedule(0.01, 5, "exp", "decreasing")
        elif use_soft_smoothing:
            smoothing = SmoothingSchedule(10, 5, "exp", "increasing")
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
