import numpy as np
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


class PlanarPoseTrajPublisher(LeafSystem):
    def __init__(
        self,
        traj: PlanarPushingTrajectory,
        delay_before_start: float = 10,
    ):
        super().__init__()
        self.traj = traj
        self.delay = delay_before_start

        self.DeclareAbstractOutputPort(
            "pusher_planar_pose",
            lambda: AbstractValue.Make(PlanarPose(x=0, y=0, theta=0)),
            self.DoCalcPusherPoseOutput,
        )

        self.DeclareAbstractOutputPort(
            "slider_planar_pose",
            lambda: AbstractValue.Make(PlanarPose(x=0, y=0, theta=0)),
            self.DoCalcSliderPoseOutput,
        )

        self.DeclareAbstractOutputPort(
            "slider_theta_dot",
            lambda: AbstractValue.Make(float),
            self.DoCalcSliderVelOutput,
        )

        self.DeclareAbstractOutputPort(
            "contact_mode",
            lambda: AbstractValue.Make(PlanarPushingContactMode(0)),
            self.DoCalcModeOuput,
        )

    def get_rel_t(self, t: float) -> float:
        return t - self.delay

    def _calc_pusher_pose(self, t: float) -> PlanarPose:
        p_c_W = self.traj.get_value(self.get_rel_t(t), "p_c_W")

        # Avoid typing error
        assert isinstance(p_c_W, type(np.array([])))

        planar_pose = PlanarPose(p_c_W[0].item(), p_c_W[1].item(), theta=0)
        return planar_pose

    def DoCalcPusherPoseOutput(self, context: Context, output):
        curr_t = context.get_time()
        end_effector_pose = self._calc_pusher_pose(curr_t)
        output.set_value(end_effector_pose)

    def _calc_slider_pose(self, t: float) -> PlanarPose:
        p_WB = self.traj.get_value(self.get_rel_t(t), "p_WB")
        theta = self.traj.get_value(self.get_rel_t(t), "theta")

        # Avoid typing error
        assert isinstance(p_WB, type(np.array([])))
        assert isinstance(theta, float)

        planar_pose = PlanarPose(p_WB[0].item(), p_WB[1].item(), theta)
        return planar_pose

    def DoCalcSliderPoseOutput(self, context: Context, output):
        curr_t = context.get_time()
        slider_pose = self._calc_slider_pose(curr_t)
        output.set_value(slider_pose)

    def _calc_slider_vel(self, t: float) -> float:
        theta_dot = self.traj.get_value(self.get_rel_t(t), "theta_dot")

        # Avoid typing error
        assert isinstance(theta_dot, float)

        return theta_dot

    def DoCalcSliderVelOutput(self, context: Context, output):
        curr_t = context.get_time()
        slider_vel = self._calc_slider_vel(curr_t)
        output.set_value(slider_vel)

    def DoCalcModeOuput(self, context: Context, output):
        curr_t = context.get_time()
        mode = self.traj.get_mode(self.get_rel_t(curr_t))
        output.set_value(mode)
