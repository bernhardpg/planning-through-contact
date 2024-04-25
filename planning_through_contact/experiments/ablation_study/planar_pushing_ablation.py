import fnmatch
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
from tqdm import tqdm

from planning_through_contact.planning.planar.planar_plan_config import (
    PlanarPlanConfig,
    PlanarSolverParams,
)
from planning_through_contact.planning.planar.utils import (
    SingleRunResult,
    do_one_run,
    sample_random_plan,
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
    for root, _, files in sorted_walk(directory):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                matches.append(os.path.join(root, name))
    return matches


@dataclass
class AblationStudy:
    results: List[SingleRunResult]

    @property
    def thetas(self) -> List[float]:
        return [res.start_and_goal.slider_initial_pose.theta for res in self.results]

    @property
    def relaxed_mean_determinants(self) -> List[float | None]:
        return [res.relaxed_mean_determinant for res in self.results]

    @property
    def rounded_mean_determinants(self) -> List[float | None]:
        return [res.rounded_mean_determinant for res in self.results]

    @property
    def distances(self) -> List[float]:
        return [float(res.distance) for res in self.results]

    @property
    def solve_times_gcs_relaxed(self) -> List[float]:
        return [res.relaxed_gcs_time for res in self.results if res.relaxed_gcs_success]

    @property
    def solve_times_binary_flows(self) -> List[float | None]:
        return [
            res.binary_flows_time
            for res in self.results
            if res.binary_flows_success and not res.numerical_difficulties
        ]

    @property
    def solve_times_feasible(self) -> List[float | None]:
        return [
            res.feasible_time
            for res in self.results
            if res.feasible_success and not res.numerical_difficulties
        ]

    @property
    def total_rounding_times(self) -> List[float | None]:
        return [
            res.total_rounding_time
            for res in self.results
            # if res.total_rounding_time is not None
        ]

    @property
    def optimality_gaps(self) -> List[float | None]:
        return [
            res.optimality_gap if res.optimality_gap is not None else None
            for res in self.results
        ]

    @property
    def feasible_is_success(self) -> List[bool | None]:
        return [
            res.feasible_success if not res.numerical_difficulties else False
            for res in self.results
        ]

    @property
    def binary_flows_success(self) -> List[float]:
        return [res.binary_flows_success for res in self.results]

    # @property
    # def optimality_percentages(self) -> List[float]:
    #     return [
    #         res.optimality_percentage if res.feasible_success else 0
    #         for res in self.results
    #     ]

    @property
    def binary_flows_optimality_gaps(self) -> List[float | None]:
        return [res.binary_flows_optimality_gap for res in self.results]

    # @property
    # def binary_flows_optimality_percentages(self) -> List[float]:
    #     return [
    #         res.binary_flows_optimality_percentage if res.binary_flows_success else 0
    #         for res in self.results
    #     ]

    @property
    def num_binary_flows_success(self) -> int:
        return len([r for r in self.results if r.binary_flows_success])

    @property
    def num_not_success(self) -> int:
        return len(self) - self.num_feasible_success

    @property
    def num_feasible_success(self) -> int:
        return np.sum(self.feasible_is_success)  # type: ignore

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

    def get_numerical_difficulties_idxs(self) -> List[str | None]:
        return [res.name for res in self.results if res.numerical_difficulties == True]

    def save(self, filename: str) -> None:
        with open(Path(filename), "wb") as file:
            pickle.dump(self.results, file)

    @classmethod
    def load(cls, filename: str) -> "AblationStudy":
        with open(Path(filename), "rb") as file:
            results = pickle.load(file)
        return cls(results)

    @classmethod
    def load_from_folder(
        cls, folder_name: str, num_to_load: Optional[int] = None
    ) -> "AblationStudy":
        data_files = _find_files(folder_name, pattern="solve_data.pkl")
        from natsort import natsorted

        data_files_sorted = natsorted(data_files)
        if num_to_load:
            data_files_sorted = data_files_sorted[:num_to_load]

        results = [SingleRunResult.load(filename) for filename in data_files_sorted]
        return AblationStudy(results)


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
