from typing import Callable, Dict, List, Optional

import numpy as np
import numpy.typing as npt
import pydrake.geometry.optimization as opt
from pydrake.common.value import Value
from pydrake.solvers import MathematicalProgramResult
from pydrake.systems.framework import BasicVector, Context, LeafSystem, OutputPort

from planning_through_contact.geometry.planar.face_contact import FaceContactVariables
from planning_through_contact.geometry.planar.non_collision_subgraph import (
    VertexModePair,
)
from planning_through_contact.geometry.planar.planar_pushing_path import (
    PlanarPushingPath,
)
from planning_through_contact.geometry.planar.planar_pushing_trajectory import (
    FaceContactTrajSegment,
)
from planning_through_contact.planning.planar.planar_plan_config import (
    SliderPusherSystemConfig,
)
from planning_through_contact.simulation.controllers.hybrid_mpc import HybridMpcConfig

GcsVertex = opt.GraphOfConvexSets.Vertex
GcsEdge = opt.GraphOfConvexSets.Edge


# TODO(bernhardpg): Use the PlanarPushingTrajectory class instead of the duplicated functionality


class SliderPusherTrajectoryFeeder(LeafSystem):
    def __init__(
        self,
        path: List[FaceContactVariables],
        config: Optional[HybridMpcConfig],
        dynamics_config: SliderPusherSystemConfig,
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
            FaceContactTrajSegment.from_knot_points(p, start, end, dynamics_config)
            for p, start, end in zip(path, self.start_times, self.end_times)
        ]

    def _get_traj_segment_for_time(self, t: float) -> FaceContactTrajSegment:
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
        if t >= self.end_times[-1]:
            traj = self.traj_segments[-1]
        else:
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
        end_time = curr_t + self.cfg.step_size * self.cfg.horizon
        ts = np.arange(curr_t, end_time, self.cfg.step_size)[: self.cfg.horizon]

        # Remove any trajectory pieces that are after the trajectory ends
        # if any(ts >= self.end_times[-1] + self.cfg.step_size):
        #     idx = np.where(ts >= self.end_times[-1] + self.cfg.step_size)[0][0]
        #     ts = ts[:idx]

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
