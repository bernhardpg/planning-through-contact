import pickle
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Literal

import numpy as np
import numpy.typing as npt
import pydrake.geometry.optimization as opt
from pydrake.common.value import Value
from pydrake.solvers import MathematicalProgramResult
from pydrake.systems.framework import BasicVector, Context, LeafSystem, OutputPort
from pydrake.trajectories import (
    PiecewisePolynomial,
    PiecewiseQuaternionSlerp,
    Trajectory,
)

from planning_through_contact.geometry.planar.abstract_mode import AbstractModeVariables
from planning_through_contact.geometry.planar.face_contact import FaceContactVariables
from planning_through_contact.geometry.planar.non_collision import NonCollisionVariables
from planning_through_contact.geometry.planar.non_collision_subgraph import (
    VertexModePair,
)
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.planar.planar_pushing_path import (
    PlanarPushingPath,
)
from planning_through_contact.geometry.planar.trajectory_builder import (
    OldPlanarPushingTrajectory,
    PlanarTrajectoryBuilder,
)
from planning_through_contact.geometry.utilities import from_so2_to_so3
from planning_through_contact.simulation.controllers.hybrid_mpc import HybridMpcConfig

GcsVertex = opt.GraphOfConvexSets.Vertex
GcsEdge = opt.GraphOfConvexSets.Edge


class So3TrajSegment:
    def __init__(
        self,
        # NOTE: we don't really need the entire matrix for this,
        # but keep it around until we want to extend to SO(3)
        Rs: List[npt.NDArray],
        start_time: float,
        end_time: float,
    ) -> None:
        self.Rs = Rs
        self.start_time = start_time
        self.end_time = end_time

        num_samples = len(Rs)
        if num_samples == 1:  # just a constant value throughout the traj time
            knot_point_times = np.array([start_time, end_time])
            samples = Rs * 2
        else:
            knot_point_times = np.linspace(start_time, end_time, num_samples)
            samples = Rs

        Rs_in_SO3 = [from_so2_to_so3(R) for R in samples]

        self.traj = PiecewiseQuaternionSlerp(knot_point_times, Rs_in_SO3)  # type: ignore

    def eval_theta(self, t: float) -> float:
        R = self.traj.orientation(t).rotation()
        return np.arccos(R[0, 0])

    def eval(self, t: float) -> npt.NDArray[np.float64]:
        R = self.traj.orientation(t).rotation()
        return R


@dataclass
class LinTrajSegment:
    num_dims: int
    start_time: float
    end_time: float
    knot_points: npt.NDArray[np.float64]
    traj: Trajectory

    @classmethod
    def from_knot_points(
        cls,
        knot_points: npt.NDArray[np.float64],
        start_time: float,
        end_time: float,
    ) -> "LinTrajSegment":
        if len(knot_points.shape) == 1:  # (NUM_SAMPLES, )
            knot_point_times = np.linspace(start_time, end_time, len(knot_points))
            samples = knot_points.reshape(
                (1, -1)
            )  # FirstOrderHold expects values to be two-dimensional
            num_dims = 1
        elif len(knot_points.shape) == 2:  # (NUM_DIMS, NUM_SAMPLES)
            num_dims, num_samples = knot_points.shape

            if num_samples == 1:  # just a constant value throughout the traj time
                knot_point_times = np.array([start_time, end_time])
                samples = np.repeat(knot_points, 2, axis=1)
            else:
                knot_point_times = np.linspace(start_time, end_time, num_samples)
                samples = knot_points
        else:
            raise ValueError("Invalid shape for knot points")

        traj = PiecewisePolynomial.FirstOrderHold(knot_point_times, samples)
        return cls(num_dims, start_time, end_time, knot_points, traj)

    def eval(self, t: float) -> float | npt.NDArray[np.float64]:
        if self.num_dims == 1:
            return self.traj.value(t).item()
        else:
            return self.traj.value(t)

    def make_derivative(self, derivative_order: int = 1) -> "LinTrajSegment":
        return LinTrajSegment(
            self.num_dims,
            self.start_time,
            self.end_time,
            self.knot_points,
            self.traj.MakeDerivative(derivative_order),
        )


# TODO(bernhardpg): Generalize
class PlanarPushingContactMode(Enum):
    NO_CONTACT = 0
    FACE_0 = 1
    FACE_1 = 2
    FACE_2 = 3
    FACE_3 = 4


@dataclass
class PlanarPushingTrajSegment:
    """
    A single trajectory segment for either a NonCollision of FaceContact mode
    for the general planar pushing system.
    """

    start_time: float
    end_time: float
    p_WB: LinTrajSegment
    R_WB: So3TrajSegment
    p_c_W: LinTrajSegment
    f_c_W: LinTrajSegment
    mode: PlanarPushingContactMode

    @classmethod
    def from_knot_points(
        cls, knot_points: AbstractModeVariables, start_time: float, end_time: float
    ) -> "PlanarPushingTrajSegment":
        p_WB = LinTrajSegment.from_knot_points(np.hstack(knot_points.p_WBs), start_time, end_time)  # type: ignore

        p_c_W = LinTrajSegment.from_knot_points(np.hstack(knot_points.p_c_Ws), start_time, end_time)  # type: ignore
        R_WB = So3TrajSegment(knot_points.R_WBs, start_time, end_time)  # type: ignore

        f_c_W = LinTrajSegment.from_knot_points(np.hstack(knot_points.f_c_Ws), start_time, end_time)  # type: ignore

        if isinstance(knot_points, NonCollisionVariables):
            mode = PlanarPushingContactMode.NO_CONTACT
        else:  # FaceContactVariables
            mode = PlanarPushingContactMode(1 + knot_points.contact_location.idx)

        return cls(start_time, end_time, p_WB, R_WB, p_c_W, f_c_W, mode)


class PlanarPushingTrajectory:
    def __init__(
        self,
        path_knot_points: List[AbstractModeVariables],
    ) -> None:
        self.path_knot_points = path_knot_points
        time_in_modes = [knot_points.time_in_mode for knot_points in path_knot_points]
        start_and_end_times = np.concatenate(([0], np.cumsum(time_in_modes)))
        self.start_times = start_and_end_times[:-1]
        self.end_times = start_and_end_times[1:]
        self.traj_segments = [
            PlanarPushingTrajSegment.from_knot_points(p, start, end)
            for p, start, end in zip(path_knot_points, self.start_times, self.end_times)
        ]

    def _get_traj_segment_for_time(self, t: float) -> PlanarPushingTrajSegment:
        idx_of_curr_segment = np.where(t <= self.end_times)[0][0]
        return self.traj_segments[idx_of_curr_segment]

    def _t_or_end_time(self, t: float) -> float:
        if t > self.end_times[-1]:
            return self.end_times[
                -1
            ]  # repeat last element when we want trajectory after end time
        else:
            return t

    def get_value(
        self,
        t: float,
        traj_to_get: Literal["p_WB", "R_WB", "p_c_W", "f_c_W", "theta"],
    ) -> npt.NDArray[np.float64] | float:
        t = self._t_or_end_time(t)
        traj = self._get_traj_segment_for_time(t)
        if traj_to_get == "p_WB":
            val = traj.p_WB.eval(t)
        elif traj_to_get == "R_WB":
            val = traj.R_WB.eval(t)
        elif traj_to_get == "p_c_W":
            val = traj.p_c_W.eval(t)
        elif traj_to_get == "f_c_W":
            val = traj.f_c_W.eval(t)
        elif traj_to_get == "theta":
            val = traj.R_WB.eval_theta(t)

        return val

    def get_mode(self, t: float) -> PlanarPushingContactMode:
        t = self._t_or_end_time(t)
        traj = self._get_traj_segment_for_time(t)
        return traj.mode

    def get_box_planar_pose(self, t) -> PlanarPose:
        p_WB = self.get_value(t, "p_WB")
        theta = self.get_value(t, "theta")

        # avoid typing errors
        assert isinstance(p_WB, type(np.array([])))
        assert isinstance(theta, float)

        planar_pose = PlanarPose(p_WB[0, 0], p_WB[1, 0], theta)
        return planar_pose

    @classmethod
    def from_result(
        cls,
        result: MathematicalProgramResult,
        gcs: opt.GraphOfConvexSets,
        source_vertex: GcsVertex,
        target_vertex: GcsVertex,
        pairs: Dict[str, VertexModePair],
        round_solution: bool = False,
    ):
        path = PlanarPushingPath.from_result(
            gcs, result, source_vertex, target_vertex, pairs
        )
        if round_solution:
            return cls(path.get_rounded_vars())
        else:
            return cls(path.get_vars())

    def save(self, filename: str) -> None:
        with open(Path(filename), "wb") as file:
            # NOTE: We save the path knot points, not this object, as some Drake objects are not serializable
            pickle.dump(self.path_knot_points, file)

    @classmethod
    def load(cls, filename: str) -> "PlanarPushingTrajectory":
        with open(Path(filename), "rb") as file:
            var_path = pickle.load(file)

            return cls(var_path)

    # TODO(bernhardpg): Remove
    def to_old_format(self) -> OldPlanarPushingTrajectory:
        return PlanarTrajectoryBuilder(self.path_knot_points).get_trajectory(
            interpolate=True
        )

    @property
    def end_time(self) -> float:
        start_time = self.traj_segments[0].start_time
        return start_time + sum([seg.end_time for seg in self.traj_segments])

    @property
    def target_planar_pose(self) -> PlanarPose:
        return self.get_box_planar_pose(self.end_time)

    @property
    def initial_planar_pose(self) -> PlanarPose:
        start_time = self.traj_segments[0].start_time
        return self.get_box_planar_pose(start_time)
