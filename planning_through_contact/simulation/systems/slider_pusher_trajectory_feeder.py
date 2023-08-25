from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

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
from planning_through_contact.geometry.planar.planar_pushing_trajectory import (
    LinTrajSegment,
    So3TrajSegment,
)
from planning_through_contact.geometry.utilities import from_so2_to_so3
from planning_through_contact.simulation.controllers.hybrid_mpc import HybridMpcConfig

GcsVertex = opt.GraphOfConvexSets.Vertex
GcsEdge = opt.GraphOfConvexSets.Edge


# TODO(bernhardpg): Consolidate this class with PlanarPushingTrajSegment
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


# TODO(bernhardpg): Use the PlanarPushingTrajectory class instead of the duplicated functionality


class SliderPusherTrajectoryFeeder(LeafSystem):
    def __init__(
        self,
        path: List[FaceContactVariables],
        config: Optional[HybridMpcConfig] = None,
    ) -> None:
        super().__init__()

        NUM_STATE_VARS = 4
        self.DeclareVectorOutputPort("state", NUM_STATE_VARS, self.CalcStateOutput)

        NUM_INPUT_VARS = 3
        self.DeclareVectorOutputPort("control", NUM_INPUT_VARS, self.CalcControlOutput)

        if config:
            self.cfg = config
            self.DeclareAbstractOutputPort(
                "state_traj",
                alloc=lambda: Value([np.array([])]),
                calc=self.CalcStateTrajOutput,  # type: ignore
            )
            self.DeclareAbstractOutputPort(
                "control_traj",
                alloc=lambda: Value([np.array([])]),
                calc=self.CalcControlTrajOutput,  # type: ignore
            )

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
            t = self.end_times[
                -1
            ]  # repeat last element when we want trajectory after end time
        traj = self._get_traj_segment_for_time(t)
        state = traj.eval_state(t)
        return state

    def get_control(self, t: float) -> npt.NDArray[np.float64]:
        if t > self.end_times[-1]:
            t = self.end_times[
                -1
            ]  # repeat last element when we want trajectory after end time
        traj = self._get_traj_segment_for_time(t)
        control = traj.eval_control(t)
        return control

    def get_state_feedforward_port(self) -> OutputPort:
        return self.GetOutputPort("state")

    def get_control_feedforward_port(self) -> OutputPort:
        return self.GetOutputPort("control")

    def CalcStateOutput(self, context: Context, output: BasicVector):
        state = self.get_state(context.get_time())
        output.SetFromVector(state)

    def CalcControlOutput(self, context: Context, output: BasicVector):
        control = self.get_control(context.get_time())
        output.SetFromVector(control)

    def _get_traj(
        self,
        curr_t: float,
        func: Callable[[float], npt.NDArray[np.float64]],
    ) -> List[npt.NDArray[np.float64]]:
        ts = np.arange(
            curr_t, curr_t + self.cfg.step_size * self.cfg.horizon, self.cfg.step_size
        )[: self.cfg.horizon]

        assert len(ts) == self.cfg.horizon

        traj = [func(t) for t in ts]
        return traj

    def CalcStateTrajOutput(self, context: Context, output):
        curr_t = context.get_time()
        state_traj = self._get_traj(curr_t, self.get_state)
        output.set_value(state_traj)

    def CalcControlTrajOutput(self, context: Context, output):
        curr_t = context.get_time()
        control_traj = self._get_traj(curr_t, self.get_control)
        output.set_value(control_traj)

    def get_state_traj_feedforward_port(self) -> OutputPort:
        return self.GetOutputPort("state_traj")

    def get_control_traj_feedforward_port(self) -> OutputPort:
        return self.GetOutputPort("control_traj")

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
