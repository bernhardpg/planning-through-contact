import numpy as np
from pydrake.common.value import AbstractValue
from pydrake.math import RigidTransform
from pydrake.systems.framework import Context, DiagramBuilder, InputPort, LeafSystem

from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.planar.planar_pushing_trajectory import (
    PlanarPushingTrajectory,
)


class PusherPosePublisher(LeafSystem):
    def __init__(
        self,
        traj: PlanarPushingTrajectory,
        z_dist_to_table: float = 0.5,
        delay_before_start: float = 10,
    ):
        super().__init__()
        self.traj = traj
        self.z_dist = z_dist_to_table
        self.delay = delay_before_start

        self.DeclareAbstractOutputPort(
            "pose", lambda: AbstractValue.Make(RigidTransform()), self.DoCalcOutput
        )

    def _calc_pose(self, t: float) -> RigidTransform:
        p_c_W = self.traj.get_value(t - self.delay, "p_c_W")
        assert isinstance(p_c_W, type(np.array([])))

        planar_pose = PlanarPose(p_c_W[0].item(), p_c_W[1].item(), theta=0)
        pose = planar_pose.to_pose(z_value=self.z_dist)
        return pose

    def DoCalcOutput(self, context: Context, output):
        curr_t = context.get_time()
        end_effector_pose = self._calc_pose(curr_t)
        output.set_value(end_effector_pose)

    @classmethod
    def add_to_builder(
        cls,
        builder: DiagramBuilder,
        traj: PlanarPushingTrajectory,
        delay: float,
        pose_input_port: InputPort,
        z_dist_buffer: float = 0.03,
    ) -> "PusherPosePublisher":
        pusher_pose_pub = builder.AddNamedSystem(
            "PusherPosePublisher",
            cls(traj, z_dist_buffer, delay_before_start=delay),
        )
        builder.Connect(
            pusher_pose_pub.get_output_port(),
            pose_input_port,
        )
        return pusher_pose_pub
