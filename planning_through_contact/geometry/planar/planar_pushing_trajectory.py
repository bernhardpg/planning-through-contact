import pickle
from dataclasses import dataclass
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
from planning_through_contact.geometry.planar.non_collision_subgraph import (
    VertexModePair,
)
from planning_through_contact.geometry.planar.planar_pushing_path import (
    PlanarPushingPath,
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


# TODO(bernhardpg): Consolidate this class with the class below?
@dataclass
class SliderPusherTrajSegment:
    """
    A single trajectory segment for a FaceContact mode
    for the slider pusher planar pushing system, which
    only includes the state x = [p_WB_x, p_WB_y, theta, lam]^T
    and the input u = [c_n, c_f, lam_dot]^T
    """

    start_time: float
    end_time: float
    p_WB_x: LinTrajSegment
    p_WB_y: LinTrajSegment
    R_WB: So3TrajSegment
    lam: LinTrajSegment
    c_n: LinTrajSegment
    c_f: LinTrajSegment
    lam_dot: LinTrajSegment

    @classmethod
    def from_knot_points(
        cls, knot_points: FaceContactVariables, start_time: float, end_time: float
    ) -> "SliderPusherTrajSegment":
        p_WB_x = LinTrajSegment.from_knot_points(knot_points.p_WB_xs, start_time, end_time)  # type: ignore
        p_WB_y = LinTrajSegment.from_knot_points(knot_points.p_WB_ys, start_time, end_time)  # type: ignore
        lam = LinTrajSegment.from_knot_points(knot_points.lams, start_time, end_time)  # type: ignore
        R_WB = So3TrajSegment(knot_points.R_WBs, start_time, end_time)  # type: ignore

        c_n = LinTrajSegment.from_knot_points(knot_points.normal_forces, start_time, end_time)  # type: ignore
        c_f = LinTrajSegment.from_knot_points(knot_points.friction_forces, start_time, end_time)  # type: ignore
        lam_dot = lam.make_derivative()

        return cls(start_time, end_time, p_WB_x, p_WB_y, R_WB, lam, c_n, c_f, lam_dot)

    def eval_state(self, t: float) -> npt.NDArray[np.float64]:
        return np.array(
            [
                self.p_WB_x.eval(t),
                self.p_WB_y.eval(t),
                self.R_WB.eval_theta(t),
                self.lam.eval(t),
            ]
        )

    def eval_control(self, t: float) -> npt.NDArray[np.float64]:
        return np.array([self.c_n.eval(t), self.c_f.eval(t), self.lam_dot.eval(t)])


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

    @classmethod
    def from_knot_points(
        cls, knot_points: AbstractModeVariables, start_time: float, end_time: float
    ) -> "PlanarPushingTrajSegment":
        p_WB = LinTrajSegment.from_knot_points(np.hstack(knot_points.p_WBs), start_time, end_time)  # type: ignore

        p_c_W = LinTrajSegment.from_knot_points(np.hstack(knot_points.p_c_Ws), start_time, end_time)  # type: ignore
        R_WB = So3TrajSegment(knot_points.R_WBs, start_time, end_time)  # type: ignore

        f_c_W = LinTrajSegment.from_knot_points(np.hstack(knot_points.f_c_Ws), start_time, end_time)  # type: ignore

        return cls(start_time, end_time, p_WB, R_WB, p_c_W, f_c_W)


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
        self, t: float, traj_to_get: Literal["p_WB", "R_WB", "p_c_W", "f_c_W"]
    ) -> npt.NDArray[np.float64]:
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

        # avoid typing error, as the trajs can return either float or np.array
        assert isinstance(val, type(np.array([])))
        return val

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
