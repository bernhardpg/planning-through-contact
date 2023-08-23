import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import numpy.typing as npt
import pydrake.geometry.optimization as opt
from pydrake.solvers import MathematicalProgramResult
from pydrake.systems.primitives import VectorLog
from pydrake.trajectories import PiecewisePolynomial, PiecewiseQuaternionSlerp

from planning_through_contact.geometry.planar.abstract_mode import AbstractModeVariables
from planning_through_contact.geometry.planar.non_collision_subgraph import (
    VertexModePair,
)
from planning_through_contact.geometry.planar.planar_pushing_path import (
    PlanarPushingPath,
)
from planning_through_contact.geometry.utilities import from_so2_to_so3

GcsVertex = opt.GraphOfConvexSets.Vertex
GcsEdge = opt.GraphOfConvexSets.Edge


@dataclass
class PlanarPushingTrajectory:
    dt: float
    R_WB: List[npt.NDArray[np.float64]]  # [(2,2) x traj_length]
    p_WB: npt.NDArray[np.float64]  # (2, traj_length)
    p_c_W: npt.NDArray[np.float64]  # (2, traj_length)
    f_c_W: npt.NDArray[np.float64]  # (2, traj_length)
    p_c_B: npt.NDArray[np.float64]  # (2, traj_length)

    def __post_init__(self) -> None:
        all_traj_lenths = np.array(
            [traj.shape[1] for traj in (self.p_WB, self.p_c_W, self.f_c_W)]
            + [len(self.R_WB)]
        )
        traj_lengths_equal = np.all(all_traj_lenths == all_traj_lenths[0])
        if not traj_lengths_equal:
            raise ValueError("Trajectories are not of equal length.")

    @property
    def N(self) -> int:
        return self.p_WB.shape[1]

    def save(self, filename: str) -> None:
        with open(Path(filename), "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename: str) -> "PlanarPushingTrajectory":
        with open(Path(filename), "rb") as file:
            obj = pickle.load(file)

            return obj


class PlanarTrajectoryBuilder:
    def __init__(self, path: List[AbstractModeVariables]):
        self.path = path

    @classmethod
    def from_result(
        cls,
        result: MathematicalProgramResult,
        gcs: opt.GraphOfConvexSets,
        source_vertex: GcsVertex,
        target_vertex: GcsVertex,
        pairs: Dict[str, VertexModePair],
    ):
        path = PlanarPushingPath.from_result(
            gcs, result, source_vertex, target_vertex, pairs
        )
        return cls(path.get_vars())

    def get_trajectory(
        self,
        dt: float = 0.01,
        interpolate: bool = True,
        assert_determinants: bool = False,
        print_determinants: bool = False,
    ) -> PlanarPushingTrajectory:
        dets = np.array([np.linalg.det(R) for p in self.path for R in p.R_WBs])
        if not all(np.isclose(dets, np.ones(dets.shape), atol=1e-04)):
            if assert_determinants:
                raise ValueError(f"Rotations do not have determinant 1: \n{dets}")
            if print_determinants:
                print("Rotations do not have determinant 1:")
                print(dets)

        if interpolate:
            R_WB = sum(
                [
                    self.interpolate_so2_using_slerp(p.R_WBs, 0, p.time_in_mode, dt)
                    for p in self.path
                ],
                [],  # merge all of the lists to one
            )
            p_WB = np.vstack(
                [self._get_traj_by_interpolation(p.p_WBs, dt, p.time_in_mode) for p in self.path]  # type: ignore
            ).T
            p_c_W = np.vstack(
                [
                    self._get_traj_by_interpolation(p.p_c_Ws, dt, p.time_in_mode)
                    for p in self.path
                ]
            ).T
            f_c_W = np.vstack(
                [
                    self._get_traj_by_interpolation(p.f_c_Ws, dt, p.time_in_mode)
                    for p in self.path
                ]
            ).T
            p_c_B = np.vstack(
                [
                    self._get_traj_by_interpolation(p.p_c_Bs, dt, p.time_in_mode)
                    for p in self.path
                ]
            ).T
        else:
            R_WB = sum(
                [p.R_WBs for p in self.path],
                [],  # merge all of the lists to one
            )
            p_WB = np.hstack(sum([p.p_WBs for p in self.path], []))
            p_c_W = np.hstack(sum([p.p_c_Ws for p in self.path], []))
            f_c_W = np.hstack(sum([p.f_c_Ws for p in self.path], []))
            p_c_B = np.hstack(sum([p.p_c_Bs for p in self.path], []))

            # Fixed dt when replaying knot points
            dt = 0.8

        return PlanarPushingTrajectory(dt, R_WB, p_WB, p_c_W, f_c_W, p_c_B)

    def _get_traj_by_interpolation(
        self,
        point_sequence: List[npt.NDArray[np.float64]],
        dt: float,
        time_in_mode: float,
    ) -> npt.NDArray[np.float64]:  # (N, 2)
        knot_points = np.hstack(point_sequence)  # (2, num_knot_points)
        return self.interpolate_w_first_order_hold(knot_points.T, 0, time_in_mode, dt)

    @staticmethod
    def interpolate_so2_using_slerp(
        Rs: List[npt.NDArray[np.float64]],
        start_time: float,
        end_time: float,
        dt: float,
    ) -> List[npt.NDArray[np.float64]]:
        """
        Assumes evenly spaced knot points R_matrices.

        @return: trajectory evaluated evenly at every dt-th step, starting at start_time and ending at specified end_time.
        """

        Rs_in_SO3 = [from_so2_to_so3(R) for R in Rs]

        # repeat the same rotation for one knot point
        if len(Rs_in_SO3) == 1:
            num_times = int(np.ceil(end_time + dt - start_time) / dt)
            R_in_SO2 = Rs_in_SO3[0][0:2, 0:2]
            R_traj_in_SO2 = [R_in_SO2] * num_times
            return R_traj_in_SO2
        else:
            knot_point_times = np.linspace(start_time, end_time, len(Rs))
            quat_slerp_traj = PiecewiseQuaternionSlerp(knot_point_times, Rs_in_SO3)  # type: ignore

            traj_times = np.arange(start_time, end_time + dt, dt)
            R_traj_in_SO2 = [
                quat_slerp_traj.orientation(t).rotation()[0:2, 0:2] for t in traj_times
            ]

            return R_traj_in_SO2

    @staticmethod
    def interpolate_w_first_order_hold(
        values: npt.NDArray[np.float64],  # (NUM_SAMPLES, NUM_DIMS)
        start_time: float,
        end_time: float,
        dt: float,
    ) -> npt.NDArray[np.float64]:  # (traj_length, num_dims)
        """
        Assumes evenly spaced knot points.

        @return: trajectory evaluated evenly at every dt-th step, starting at start_time and ending at specified end_time.
        """

        if len(values) == 1:
            num_times = int(np.ceil(end_time + dt - start_time) / dt)
            traj = values.repeat(num_times, axis=0)  # (num_times, num_dims)
            return traj
        else:
            knot_point_times = np.linspace(start_time, end_time, len(values))

            # Drake expects the values to be (NUM_DIMS, NUM_SAMPLES)
            first_order_hold = PiecewisePolynomial.FirstOrderHold(
                knot_point_times, values.T
            )
            traj_times = np.arange(start_time, end_time + dt, dt)
            traj = np.hstack(
                [first_order_hold.value(t) for t in traj_times]
            ).T  # (traj_length, num_dims)

        return traj
