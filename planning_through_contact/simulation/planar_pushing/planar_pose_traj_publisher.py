from typing import Callable, List, TypeVar

import numpy as np
import numpy.typing as npt
from pydrake.common.value import AbstractValue
from pydrake.systems.framework import Context, LeafSystem

from planning_through_contact.geometry.planar.planar_pose import (
    PlanarPose,
    PlanarVelocity,
)
from planning_through_contact.geometry.planar.planar_pushing_trajectory import (
    PlanarPushingContactMode,
    PlanarPushingTrajectory,
)
from planning_through_contact.simulation.controllers.hybrid_mpc import HybridMpcConfig


class PlanarPoseTrajPublisher(LeafSystem):
    def __init__(
        self,
        traj: PlanarPushingTrajectory,
        mpc_config: HybridMpcConfig,
        delay_before_start: float = 10,
    ):
        super().__init__()
        self.traj = traj
        self.delay = delay_before_start
        self.mpc_config = mpc_config

        self.DeclareAbstractOutputPort(
            "pusher_planar_pose_traj",
            lambda: AbstractValue.Make([PlanarPose(x=0, y=0, theta=0)]),
            self.DoCalcPusherPoseTrajOutput,
        )

        self.DeclareAbstractOutputPort(
            "slider_planar_pose_traj",
            lambda: AbstractValue.Make([PlanarPose(x=0, y=0, theta=0)]),
            self.DoCalcSliderPoseTrajOutput,
        )

        self.DeclareAbstractOutputPort(
            "contact_force_traj",
            lambda: AbstractValue.Make([np.array([])]),
            self.DoCalcForceTrajOutput,
        )

        self.DeclareAbstractOutputPort(
            "contact_mode_traj",
            lambda: AbstractValue.Make([PlanarPushingContactMode(0)]),
            self.DoCalcModeTrajOutput,
        )

        self.DeclareVectorOutputPort(
            "desired_pusher_planar_pose_vector",
            3,
            self.DoCalcDesiredPusherPlanarPoseVectorOutput,
        )

        self.DeclareVectorOutputPort(
            "desired_slider_planar_pose_vector",
            3,
            self.DoCalcDesiredSliderPlanarPoseVectorOutput,
        )

    def _get_rel_t(self, t: float) -> float:
        return t - self.delay

    T = TypeVar("T")

    def _get_traj(self, curr_t: float, func: Callable[[float], T]) -> List[T]:
        h = self.mpc_config.step_size
        N = self.mpc_config.horizon

        ts = np.arange(curr_t, curr_t + h * N, h)[:N]
        assert len(ts) == N

        traj = [func(t) for t in ts]
        return traj

    def _calc_pusher_pose(self, t: float) -> PlanarPose:
        p_WP = self.traj.get_value(t, "p_WP")

        # Avoid typing error
        assert isinstance(p_WP, type(np.array([])))

        planar_pose = PlanarPose(p_WP[0].item(), p_WP[1].item(), theta=0)
        return planar_pose

    def DoCalcPusherPoseTrajOutput(self, context: Context, output):
        curr_t = context.get_time()
        pusher_traj = self._get_traj(self._get_rel_t(curr_t), self._calc_pusher_pose)
        output.set_value(pusher_traj)

    def _calc_slider_pose(self, t: float) -> PlanarPose:
        p_WB = self.traj.get_value(t, "p_WB")
        theta = self.traj.get_value(t, "theta")

        # Avoid typing error
        assert isinstance(p_WB, type(np.array([])))
        assert isinstance(theta, float)

        planar_pose = PlanarPose(p_WB[0].item(), p_WB[1].item(), theta)
        return planar_pose

    def DoCalcSliderPoseTrajOutput(self, context: Context, output):
        curr_t = context.get_time()
        slider_traj = self._get_traj(self._get_rel_t(curr_t), self._calc_slider_pose)
        output.set_value(slider_traj)

    def DoCalcForceTrajOutput(self, context: Context, output) -> None:
        curr_t = context.get_time()
        force_traj = self._get_traj(
            self._get_rel_t(curr_t), lambda t: self.traj.get_value(t, "f_c_W")
        )
        output.set_value(force_traj)

    def DoCalcModeTrajOutput(self, context: Context, output):
        curr_t = context.get_time()
        mode_traj = self._get_traj(
            self._get_rel_t(curr_t), lambda t: self.traj.get_mode(t)
        )
        output.set_value(mode_traj)

    def DoCalcDesiredPusherPlanarPoseVectorOutput(self, context: Context, output):
        curr_t = context.get_time()
        pusher_pose = self._calc_pusher_pose(self._get_rel_t(curr_t))
        output.SetFromVector(pusher_pose.vector())

    def DoCalcDesiredSliderPlanarPoseVectorOutput(self, context: Context, output):
        curr_t = context.get_time()
        slider_pose = self._calc_slider_pose(self._get_rel_t(curr_t))
        output.SetFromVector(slider_pose.vector())
