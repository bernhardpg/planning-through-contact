import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional

import numpy as np
from tqdm import tqdm

from planning_through_contact.experiments.utils import (
    get_default_plan_config,
    get_default_solver_params,
    sample_random_plan,
)
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.planar.planar_pushing_path import (
    PlanarPushingPath,
)
from planning_through_contact.planning.planar.planar_plan_config import (
    PlanarPlanConfig,
    PlanarPushingStartAndGoal,
    PlanarSolverParams,
)
from planning_through_contact.planning.planar.planar_pushing_planner import (
    PlanarPushingPlanner,
)
from planning_through_contact.visualize.planar_pushing import (
    visualize_planar_pushing_start_and_goal,
)


@dataclass
class SingleRunResult:
    sdp_cost: float
    rounded_cost: float
    relaxed_cost: float
    sdp_elapsed_time: float
    rounding_elapsed_time: float
    relaxed_elapsed_time: float
    sdp_is_success: bool
    relaxed_is_success: bool
    rounded_is_success: bool
    start_and_goal: PlanarPushingStartAndGoal

    @property
    def optimality_gap(self) -> float:
        return (self.relaxed_cost / self.rounded_cost) * 100

    @property
    def sdp_optimality_gap(self) -> float:
        return (self.relaxed_cost / self.sdp_cost) * 100

    @property
    def distance(self) -> float:
        start = self.start_and_goal.slider_initial_pose.pos()
        end = self.start_and_goal.slider_target_pose.pos()
        dist: float = np.linalg.norm(start - end)
        return dist


@dataclass
class AblationStudy:
    results: List[SingleRunResult]

    @property
    def thetas(self) -> List[float]:
        return [res.start_and_goal.slider_initial_pose.theta for res in self.results]

    @property
    def distances(self) -> List[float]:
        return [res.distance for res in self.results]

    @property
    def optimality_gaps(self) -> List[float]:
        return [res.optimality_gap for res in self.results]

    @property
    def sdp_optimality_gaps(self) -> List[float]:
        return [res.sdp_optimality_gap for res in self.results]

    def save(self, filename: str) -> None:
        with open(Path(filename), "wb") as file:
            pickle.dump(self.results, file)

    @classmethod
    def load(cls, filename: str) -> "AblationStudy":
        with open(Path(filename), "rb") as file:
            results = pickle.load(file)
        return cls(results)


def do_one_run(
    plan_config: PlanarPlanConfig,
    solver_params: PlanarSolverParams,
    start_and_goal: PlanarPushingStartAndGoal,
):
    plan_config.start_and_goal = start_and_goal

    planner = PlanarPushingPlanner(plan_config)
    planner.formulate_problem()

    # Store this value as we need to set it to 0 to run GCS without rounding!
    max_rounded_paths = solver_params.gcs_max_rounded_paths

    start_time = time.time()
    solver_params.gcs_max_rounded_paths = 0
    relaxed_result = planner._solve(solver_params)
    relaxed_elapsed_time = time.time() - start_time
    relaxed_cost = relaxed_result.get_optimal_cost()

    solver_params.gcs_max_rounded_paths = max_rounded_paths
    start_time = time.time()
    sdp_result = planner._solve(solver_params)
    sdp_elapsed_time = time.time() - start_time
    sdp_cost = sdp_result.get_optimal_cost()

    if not sdp_result.is_success():
        visualize_planar_pushing_start_and_goal(
            plan_config.dynamics_config.slider.geometry,
            plan_config.dynamics_config.pusher_radius,
            start_and_goal,
            # show=True,
            save=True,
            filename=f"infeasible_trajectory",
        )

    assert planner.source is not None  # avoid typing errors
    assert planner.target is not None  # avoid typing errors
    path = PlanarPushingPath.from_result(
        planner.gcs,
        sdp_result,
        planner.source.vertex,
        planner.target.vertex,
        planner._get_all_vertex_mode_pairs(),
    )
    start_time = time.time()
    rounding_result = path._do_nonlinear_rounding(solver_params)
    rounding_elapsed_time = time.time() - start_time
    rounded_cost = rounding_result.get_optimal_cost()

    return SingleRunResult(
        sdp_cost,
        rounded_cost,
        relaxed_cost,
        sdp_elapsed_time,
        rounding_elapsed_time,
        relaxed_elapsed_time,
        sdp_result.is_success(),
        relaxed_result.is_success(),
        rounding_result.is_success(),
        start_and_goal,
    )


def run_ablation(
    plan_config: PlanarPlanConfig,
    solver_params: PlanarSolverParams,
    num_experiments: int,
    filename: Optional[str] = None,
) -> None:
    # We want the plans to always be the same
    np.random.seed(999)

    solver_params.save_solver_output = True

    results = []
    for _ in tqdm(range(num_experiments)):
        start_and_goal = sample_random_plan()
        result = do_one_run(plan_config, solver_params, start_and_goal)
        results.append(result)

    study = AblationStudy(results)
    if filename is not None:
        study.save(filename)


def run_ablation_with_default_config(
    slider_type: Literal["box", "sugar_box", "tee"],
    num_experiments: int,
    filename: Optional[str] = None,
) -> None:
    config = get_default_plan_config()
    solver_params = get_default_solver_params()
    run_ablation(config, solver_params, num_experiments, filename)
