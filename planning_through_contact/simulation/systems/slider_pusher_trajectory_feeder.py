from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import numpy.typing as npt
import pydrake.geometry.optimization as opt
from pydrake.solvers import MathematicalProgramResult
from pydrake.systems.framework import BasicVector, Context, LeafSystem, OutputPort
from pydrake.trajectories import (
    PiecewisePolynomial,
    PiecewiseQuaternionSlerp,
    Trajectory,
)

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


class So2TrajSegment:
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

        Rs_in_SO3 = [from_so2_to_so3(R) for R in Rs]
        knot_point_times = np.linspace(start_time, end_time, len(Rs))
        self.traj = PiecewiseQuaternionSlerp(knot_point_times, Rs_in_SO3)  # type: ignore

    def eval(self, t: float) -> float:
        R = self.traj.orientation(t).rotation()
        return np.arccos(R[0, 0])


@dataclass
class LinTrajSegment:
    knot_points: npt.NDArray[np.float64]
    start_time: float
    end_time: float
    traj: Trajectory

    @classmethod
    def from_knot_points(
        cls,
        knot_points: npt.NDArray[np.float64],  # (NUM_SAMPLES, )
        start_time: float,
        end_time: float,
    ) -> "LinTrajSegment":
        knot_point_times = np.linspace(start_time, end_time, len(knot_points))
        traj = PiecewisePolynomial.FirstOrderHold(
            knot_point_times,
            knot_points.reshape(
                (1, -1)
            ),  # FirstOrderHold expects values to be two-dimensional
        )
        return cls(knot_points, start_time, end_time, traj)

    def eval(self, t: float) -> None:  # TODO
        return self.traj.value(t).item()

    def make_derivative(self, derivative_order: int = 1) -> "LinTrajSegment":
        return LinTrajSegment(
            self.knot_points,
            self.start_time,
            self.end_time,
            self.traj.MakeDerivative(derivative_order),
        )


@dataclass
class SliderPusherTrajSegment:
    start_time: float
    end_time: float
    p_WB_x: LinTrajSegment
    p_WB_y: LinTrajSegment
    theta: So2TrajSegment
    lam: LinTrajSegment
    c_n: LinTrajSegment
    c_f: LinTrajSegment
    lam_dot: LinTrajSegment

    @classmethod
    def from_knot_points(
        cls, knot_points: AbstractModeVariables, start_time: float, end_time: float
    ) -> "SliderPusherTrajSegment":
        p_WB_x = LinTrajSegment.from_knot_points(knot_points.p_WB_xs, start_time, end_time)  # type: ignore
        p_WB_y = LinTrajSegment.from_knot_points(knot_points.p_WB_ys, start_time, end_time)  # type: ignore
        lam = LinTrajSegment.from_knot_points(knot_points.lams, start_time, end_time)  # type: ignore
        theta = So2TrajSegment(knot_points.R_WBs, start_time, end_time)  # type: ignore

        c_n = LinTrajSegment.from_knot_points(knot_points.normal_forces, start_time, end_time)  # type: ignore
        c_f = LinTrajSegment.from_knot_points(knot_points.friction_forces, start_time, end_time)  # type: ignore
        lam_dot = lam.make_derivative()

        return cls(start_time, end_time, p_WB_x, p_WB_y, theta, lam, c_n, c_f, lam_dot)

    def eval_state(self, t: float) -> npt.NDArray[np.float64]:
        return np.array(
            [
                self.p_WB_x.eval(t),
                self.p_WB_y.eval(t),
                self.theta.eval(t),
                self.lam.eval(t),
            ]
        )

    def eval_control(self, t: float) -> npt.NDArray[np.float64]:
        return np.array([self.c_n.eval(t), self.c_f.eval(t), self.lam_dot.eval(t)])


class SliderPusherTrajectoryFeeder(LeafSystem):
    def __init__(self, path: List[AbstractModeVariables]) -> None:
        super().__init__()

        NUM_STATE_VARS = 4
        self.DeclareVectorOutputPort("state", NUM_STATE_VARS, self.CalcStateOutput)

        NUM_INPUT_VARS = 3
        self.DeclareVectorOutputPort("control", NUM_INPUT_VARS, self.CalcControlOutput)

        time_in_modes = [knot_points.time_in_mode for knot_points in path]
        temp = np.concatenate(([0], np.cumsum(time_in_modes)))
        self.start_times = temp[:-1]
        self.end_times = temp[1:]
        self.traj_segments = [
            SliderPusherTrajSegment.from_knot_points(p, start, end)
            for p, start, end in zip(path, self.start_times, self.end_times)
        ]

    def _get_traj_segment_for_time(self, t: float) -> SliderPusherTrajSegment:
        idx_of_curr_segment = np.where(t <= self.end_times)[0][0]
        return self.traj_segments[idx_of_curr_segment]

    def get_state(self, t: float) -> npt.NDArray[np.float64]:
        if t > self.end_times[-1]:
            raise RuntimeError(
                f"Cannot get value for time {t}, last end_time is {self.end_times[-1]}"
            )
        traj = self._get_traj_segment_for_time(t)
        state = traj.eval_state(t)
        return state

    def get_control(self, t: float) -> npt.NDArray[np.float64]:
        if t > self.end_times[-1]:
            raise RuntimeError(
                f"Cannot get value for time {t}, last end_time is {self.end_times[-1]}"
            )
        traj = self._get_traj_segment_for_time(t)
        control = traj.eval_control(t)
        return control

    def get_state_feedforward_port(self) -> OutputPort:
        return self.GetOutputPort("state")

    def get_control_feedforward_port(self) -> OutputPort:
        return self.GetOutputPort("control")

    def CalcStateOutput(self, context: Context, output: BasicVector):
        output.SetFromVector(self.get_state(context.get_time()))

    def CalcControlOutput(self, context: Context, output: BasicVector):
        output.SetFromVector(self.get_control(context.get_time()))

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
        return cls(path.get_rounded_vars())
