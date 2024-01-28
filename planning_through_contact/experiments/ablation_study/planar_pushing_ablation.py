import fnmatch
import os
import pickle
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import List, Literal, Optional, Tuple

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


def _find_files(directory, pattern):
    matches = []
    for root, dirs, files in os.walk(directory):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                matches.append(os.path.join(root, name))
    return matches


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
    relaxed_mean_determinant: float
    rounded_mean_determinant: float
    start_and_goal: PlanarPushingStartAndGoal
    config: PlanarPlanConfig

    @property
    def optimality_gap(self) -> float:
        return ((self.rounded_cost - self.relaxed_cost) / self.relaxed_cost) * 100

    @property
    def sdp_optimality_gap(self) -> float:
        return ((self.sdp_cost - self.relaxed_cost) / self.sdp_cost) * 100

    @property
    def optimality_percentage(self) -> float:
        return 100 - self.optimality_gap

    @property
    def sdp_optimality_percentage(self) -> float:
        return 100 - self.optimality_gap

    @property
    def distance(self) -> float:
        start = self.start_and_goal.slider_initial_pose.pos()
        end = self.start_and_goal.slider_target_pose.pos()
        dist: float = np.linalg.norm(start - end)
        return dist

    def save(self, filename: str) -> None:
        with open(Path(filename), "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename: str) -> "SingleRunResult":
        with open(Path(filename), "rb") as file:
            return pickle.load(file)

    def __str__(self):
        # Manually add property strings
        property_strings = [
            f"optimality_gap: {self.optimality_gap}",
            f"sdp_optimality_gap: {self.sdp_optimality_gap}",
        ]

        field_strings = [
            f"{field.name}: {getattr(self, field.name)}" for field in fields(self)
        ]

        # Combine field and property strings
        all_strings = field_strings + property_strings
        return "\n".join(all_strings)

    def save_as_text(self, filename: str) -> None:
        with open(filename, "w") as file:
            file.write(str(self))


@dataclass
class AblationStudy:
    results: List[SingleRunResult]

    @property
    def thetas(self) -> List[float]:
        return [res.start_and_goal.slider_initial_pose.theta for res in self.results]

    @property
    def relaxed_mean_determinants(self) -> List[float]:
        return [res.relaxed_mean_determinant for res in self.results]

    @property
    def rounded_mean_determinants(self) -> List[float]:
        return [res.rounded_mean_determinant for res in self.results]

    @property
    def distances(self) -> List[float]:
        return [res.distance for res in self.results]

    @property
    def mean_solve_time_sdp(self) -> float:
        return np.mean([res.sdp_elapsed_time for res in self.results])

    @property
    def mean_solve_time_rounding(self) -> float:
        return np.mean([res.rounding_elapsed_time for res in self.results])

    @property
    def optimality_gaps(self) -> List[float]:
        return [res.optimality_gap for res in self.results]

    @property
    def mean_optimality_gap(self) -> List[float]:
        return np.mean(
            [res.optimality_gap for res in self.results if res.rounded_is_success]
        )

    @property
    def rounded_is_success(self) -> List[float]:
        return [res.rounded_is_success for res in self.results]

    @property
    def sdp_is_success(self) -> List[float]:
        return [res.sdp_is_success for res in self.results]

    @property
    def optimality_percentages(self) -> List[float]:
        return [
            res.optimality_percentage if res.rounded_is_success else 0
            for res in self.results
        ]

    @property
    def sdp_optimality_gaps(self) -> List[float]:
        return [res.sdp_optimality_gap for res in self.results]

    @property
    def sdp_optimality_percentages(self) -> List[float]:
        return [
            res.sdp_optimality_percentage if res.sdp_is_success else 0
            for res in self.results
        ]

    @property
    def num_success(self) -> int:
        return len([r for r in self.results if r.sdp_is_success])

    @property
    def num_not_success(self) -> int:
        return len(self) - self.num_success

    @property
    def num_rounded_success(self) -> int:
        return len([r for r in self.results if r.rounded_is_success])

    def __len__(self) -> int:
        return len(self.results)

    @property
    def num_rounded_not_success(self) -> int:
        return len(self) - self.num_rounded_success

    @property
    def percentage_success(self) -> float:
        return (self.num_success / len(self)) * 100

    @property
    def percentage_rounded_success(self) -> float:
        return (self.num_rounded_success / len(self)) * 100

    def save(self, filename: str) -> None:
        with open(Path(filename), "wb") as file:
            pickle.dump(self.results, file)

    @classmethod
    def load(cls, filename: str) -> "AblationStudy":
        with open(Path(filename), "rb") as file:
            results = pickle.load(file)
        return cls(results)

    @classmethod
    def load_from_folder(cls, folder_name: str) -> "AblationStudy":
        data_files = _find_files(folder_name, pattern="solve_data.pkl")

        results = [SingleRunResult.load(filename) for filename in data_files]
        return AblationStudy(results)


def do_one_run_get_path(
    plan_config: PlanarPlanConfig,
    solver_params: PlanarSolverParams,
    start_and_goal: PlanarPushingStartAndGoal,
) -> Tuple[SingleRunResult, Optional[PlanarPushingPath]]:
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

        return (
            SingleRunResult(
                np.inf,
                np.inf,
                relaxed_cost,
                sdp_elapsed_time,
                np.inf,
                relaxed_elapsed_time,
                sdp_result.is_success(),
                relaxed_result.is_success(),
                False,
                np.inf,
                np.inf,
                start_and_goal,
                plan_config,
            ),
            None,
        )

    assert planner.source is not None  # avoid typing errors
    assert planner.target is not None  # avoid typing errors
    path = PlanarPushingPath.from_result(
        planner.gcs,
        sdp_result,
        planner.source.vertex,
        planner.target.vertex,
        planner._get_all_vertex_mode_pairs(),
        assert_nan_values=solver_params.assert_nan_values,
    )
    start_time = time.time()
    rounding_result = path._do_nonlinear_rounding(solver_params)
    rounding_elapsed_time = time.time() - start_time
    rounded_cost = rounding_result.get_optimal_cost()

    relaxed_mean_determinant: float = np.mean(path.get_determinants())

    path.rounded_result = rounding_result
    rounded_mean_determinant: float = np.mean(path.get_determinants(rounded=True))

    return (
        SingleRunResult(
            sdp_cost,
            rounded_cost,
            relaxed_cost,
            sdp_elapsed_time,
            rounding_elapsed_time,
            relaxed_elapsed_time,
            sdp_result.is_success(),
            relaxed_result.is_success(),
            rounding_result.is_success(),
            relaxed_mean_determinant,
            rounded_mean_determinant,
            start_and_goal,
            plan_config,
        ),
        path,
    )


def do_one_run(
    plan_config: PlanarPlanConfig,
    solver_params: PlanarSolverParams,
    start_and_goal: PlanarPushingStartAndGoal,
):
    run, _ = do_one_run_get_path(plan_config, solver_params, start_and_goal)
    return run


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
    pusher_radius: float,
    integration_constant: float,
    num_experiments: int,
    arc_length_weight: Optional[float] = None,
    filename: Optional[str] = None,
) -> None:
    config = get_default_plan_config(
        slider_type, pusher_radius, integration_constant, arc_length_weight
    )
    solver_params = get_default_solver_params()
    run_ablation(config, solver_params, num_experiments, filename)
