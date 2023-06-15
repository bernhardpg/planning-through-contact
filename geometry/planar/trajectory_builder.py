from dataclasses import dataclass
from typing import List

import numpy as np
import numpy.typing as npt
from pydrake.trajectories import PiecewisePolynomial, PiecewiseQuaternionSlerp

from geometry.planar.planar_contact_modes import (
    FaceContactVariables,
    NonCollisionVariables,
)
from geometry.utilities import from_so2_to_so3


@dataclass
class PlanarTrajectory:
    dt: float
    R_WB: List[npt.NDArray[np.float64]]  # [(2,2) x traj_length]
    p_WB: npt.NDArray[np.float64]  # (2, traj_length)
    p_c_W: npt.NDArray[np.float64]  # (2, traj_length)
    f_c_W: npt.NDArray[np.float64]  # (2, traj_length)

    def __post_init__(self) -> None:
        dets = np.array([np.linalg.det(R) for R in self.R_WB])
        if not all(np.isclose(dets, np.ones(dets.shape), atol=1e-02)):
            raise ValueError("Rotations do not have determinant 1.")

    @property
    def N(self) -> int:
        return self.p_WB.shape[1]


class PlanarTrajectoryBuilder:
    def __init__(self, path: List[FaceContactVariables | NonCollisionVariables]):
        self.path = path

    def get_trajectory(
        self, dt: float = 0.01, interpolate: bool = True
    ) -> PlanarTrajectory:
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
            )
            p_c_W = np.vstack(
                [
                    self._get_traj_by_interpolation(p.p_c_Ws, dt, p.time_in_mode)
                    for p in self.path
                ]
            )
            f_c_W = np.vstack(
                [
                    self._get_traj_by_interpolation(p.f_c_Ws, dt, p.time_in_mode)
                    for p in self.path
                ]
            )
        else:
            R_WB = sum(
                [p.R_WBs for p in self.path],
                [],  # merge all of the lists to one
            )
            p_WB = np.vstack([p.p_WBs for p in self.path])
            p_c_W = np.vstack([p.p_c_Ws for p in self.path])
            f_c_W = np.vstack([p.f_c_Ws for p in self.path])

            dt = 0.8

        return PlanarTrajectory(dt, R_WB, p_WB, p_c_W, f_c_W)

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
        knot_point_times = np.linspace(start_time, end_time, len(Rs))
        quat_slerp_traj = PiecewiseQuaternionSlerp(knot_point_times, Rs_in_SO3)  # type: ignore

        traj_times = np.arange(start_time, end_time, dt)
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
    ) -> npt.NDArray[np.float64]:  # (NUM_POINTS, NUM_DIMS)
        """
        Assumes evenly spaced knot points.

        @return: trajectory evaluated evenly at every dt-th step, starting at start_time and ending at specified end_time.
        """

        knot_point_times = np.linspace(start_time, end_time, len(values))

        # Drake expects the values to be (NUM_DIMS, NUM_SAMPLES)
        first_order_hold = PiecewisePolynomial.FirstOrderHold(
            knot_point_times, values.T
        )
        traj_times = np.arange(start_time, end_time, dt)
        traj = np.hstack(
            [first_order_hold.value(t) for t in traj_times]
        ).T  # transpose to match format in rest of project

        return traj
