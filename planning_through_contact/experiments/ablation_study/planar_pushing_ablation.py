import fnmatch
import os
import pickle
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


def sorted_walk(top, topdown=True):
    dirs, nondirs = [], []
    for name in sorted(os.listdir(top)):
        (dirs if os.path.isdir(os.path.join(top, name)) else nondirs).append(name)
    if topdown:
        yield top, dirs, nondirs
    for name in dirs:
        path = os.path.join(top, name)
        for x in sorted_walk(path, topdown):
            yield x
    if not topdown:
        yield top, dirs, nondirs


def _find_files(directory, pattern):
    matches = []
    for root, dirs, files in sorted_walk(directory):
        print(root)
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                matches.append(os.path.join(root, name))
    return matches


@dataclass
class SingleRunResult:
    relaxed_gcs_cost: float
    relaxed_gcs_success: bool
    relaxed_gcs_time: float
    binary_flows_cost: float
    binary_flows_success: bool
    binary_flows_time: float
    feasible_cost: float
    feasible_success: bool
    feasible_time: Optional[float]
    relaxed_mean_determinant: float
    rounded_mean_determinant: float
    start_and_goal: PlanarPushingStartAndGoal
    config: PlanarPlanConfig
    name: Optional[str] = None

    @property
    def optimality_gap(self) -> float:
        return (
            (self.feasible_cost - self.relaxed_gcs_cost) / self.relaxed_gcs_cost
        ) * 100

    @property
    def binary_flows_optimality_gap(self) -> float:
        return (
            (self.binary_flows_cost - self.relaxed_gcs_cost) / self.relaxed_gcs_cost
        ) * 100

    @property
    def optimality_percentage(self) -> float:
        return 100 - self.optimality_gap

    @property
    def binary_flows_optimality_percentage(self) -> float:
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
            run_result = pickle.load(file)
            run_result.name = filename
            return run_result

    def __str__(self):
        # Manually add property strings
        property_strings = [
            f"optimality_gap: {self.optimality_gap}",
            f"sdp_optimality_gap: {self.binary_flows_optimality_gap}",
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
    def solve_times_binary_flows(self) -> List[float]:
        return [
            res.binary_flows_time for res in self.results if res.binary_flows_success
        ]

    @property
    def solve_times_feasible(self) -> List[float | None]:
        return [res.feasible_time for res in self.results if res.feasible_success]

    @property
    def optimality_gaps(self) -> List[float]:
        return [res.optimality_gap for res in self.results]

    @property
    def feasible_is_success(self) -> List[float]:
        return [res.feasible_success for res in self.results]

    @property
    def binary_flows_success(self) -> List[float]:
        return [res.binary_flows_success for res in self.results]

    @property
    def optimality_percentages(self) -> List[float]:
        return [
            res.optimality_percentage if res.feasible_success else 0
            for res in self.results
        ]

    @property
    def binary_flows_optimality_gaps(self) -> List[float]:
        return [res.binary_flows_optimality_gap for res in self.results]

    @property
    def binary_flows_optimality_percentages(self) -> List[float]:
        return [
            res.binary_flows_optimality_percentage if res.binary_flows_success else 0
            for res in self.results
        ]

    @property
    def num_binary_flows_success(self) -> int:
        return len([r for r in self.results if r.binary_flows_success])

    @property
    def num_not_success(self) -> int:
        return len(self) - self.num_binary_flows_success

    @property
    def num_feasible_success(self) -> int:
        return len([r for r in self.results if r.feasible_success])

    def __len__(self) -> int:
        return len(self.results)

    @property
    def num_rounded_not_success(self) -> int:
        return len(self) - self.num_feasible_success

    @property
    def percentage_binary_flows_success(self) -> float:
        return (self.num_binary_flows_success / len(self)) * 100

    @property
    def percentage_feasible_success(self) -> float:
        return (self.num_feasible_success / len(self)) * 100

    def get_infeasible_idxs(self) -> List[str | None]:
        return [res.name for res in self.results if not res.feasible_success]

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

    path = planner.plan_path(solver_params)

    assert planner.source is not None  # avoid typing errors
    assert planner.target is not None  # avoid typing errors

    relaxed_mean_determinant: float = np.mean(path.get_determinants())
    rounded_mean_determinant: float = np.mean(path.get_determinants(rounded=True))

    assert planner.relaxed_gcs_result is not None
    assert path.rounded_result is not None
    return (
        SingleRunResult(
            relaxed_gcs_cost=planner.relaxed_gcs_result.get_optimal_cost(),
            relaxed_gcs_success=planner.relaxed_gcs_result.is_success(),
            relaxed_gcs_time=planner.relaxed_gcs_result.get_solver_details().optimizer_time,  # type: ignore
            binary_flows_cost=path.relaxed_cost,
            binary_flows_success=path.result.is_success(),
            binary_flows_time=path.solve_time,
            feasible_cost=path.rounded_cost,
            feasible_success=path.rounded_result.is_success(),
            feasible_time=path.rounding_time,
            relaxed_mean_determinant=relaxed_mean_determinant,
            rounded_mean_determinant=rounded_mean_determinant,
            start_and_goal=start_and_goal,
            config=plan_config,
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
