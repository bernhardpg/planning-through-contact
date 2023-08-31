from typing import Callable, List

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
            "contact_mode",
            lambda: AbstractValue.Make(PlanarPushingContactMode(0)),
            self.DoCalcModeOuput,
        )

    def _get_rel_t(self, t: float) -> float:
        return t - self.delay

    def _get_traj(
        self,
        curr_t: float,
        func: Callable[[float], PlanarPose],
    ) -> List[PlanarPose]:
        h = self.mpc_config.step_size
        N = self.mpc_config.horizon

        ts = np.arange(curr_t, curr_t + h * N, h)[:N]
        assert len(ts) == N

        traj = [func(t) for t in ts]
        return traj

    def _calc_pusher_pose(self, t: float) -> PlanarPose:
        p_c_W = self.traj.get_value(self._get_rel_t(t), "p_c_W")

        # Avoid typing error
        assert isinstance(p_c_W, type(np.array([])))

        planar_pose = PlanarPose(p_c_W[0].item(), p_c_W[1].item(), theta=0)
        return planar_pose

    def DoCalcPusherPoseTrajOutput(self, context: Context, output):
        curr_t = context.get_time()
        pusher_traj = self._get_traj(self._get_rel_t(curr_t), self._calc_pusher_pose)
        output.set_value(pusher_traj)

    def _calc_slider_pose(self, t: float) -> PlanarPose:
        p_WB = self.traj.get_value(self._get_rel_t(t), "p_WB")
        theta = self.traj.get_value(self._get_rel_t(t), "theta")

        # Avoid typing error
        assert isinstance(p_WB, type(np.array([])))
        assert isinstance(theta, float)

        planar_pose = PlanarPose(p_WB[0].item(), p_WB[1].item(), theta)
        return planar_pose

    def DoCalcSliderPoseTrajOutput(self, context: Context, output):
        curr_t = context.get_time()
        slider_traj = self._get_traj(self._get_rel_t(curr_t), self._calc_slider_pose)
        output.set_value(slider_traj)

    def DoCalcModeOuput(self, context: Context, output):
        curr_t = context.get_time()
        mode = self.traj.get_mode(self._get_rel_t(curr_t))
        output.set_value(mode)
