import numpy as np
from pydrake.common.value import AbstractValue
from pydrake.systems.framework import Context, LeafSystem

from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.planar.planar_pushing_trajectory import (
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
            "pose",
            lambda: AbstractValue.Make(PlanarPose(x=0, y=0, theta=0)),
            self.DoCalcOutput,
        )

    def _calc_pose(self, t: float) -> PlanarPose:
        p_c_W = self.traj.get_value(t - self.delay, "p_c_W")

        # Avoid typing error
        assert isinstance(p_c_W, type(np.array([])))

        planar_pose = PlanarPose(p_c_W[0].item(), p_c_W[1].item(), theta=0)
        return planar_pose

    def DoCalcOutput(self, context: Context, output):
        curr_t = context.get_time()
        end_effector_pose = self._calc_pose(curr_t)
        output.set_value(end_effector_pose)
