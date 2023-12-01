from pydrake.systems.framework import Diagram, DiagramBuilder

from planning_through_contact.geometry.planar.planar_pushing_trajectory import (
    PlanarPushingTrajectory,
)
from planning_through_contact.geometry.rigid_body import RigidBody
from planning_through_contact.simulation.planar_pushing.planar_pushing_diagram import (
    PlanarPushingSimConfig,
)
from planning_through_contact.simulation.planar_pushing.planar_pushing_sim import (
    PlanarPoseTrajPublisher,
    PusherPoseToJointPosDiffIk,
)
from planning_through_contact.simulation.planar_pushing.pusher_pose_controller import (
    PusherPoseController,
)


class OpenLoopPushingController(Diagram):
    def __init__(
        self,
        traj: PlanarPushingTrajectory,
        slider: RigidBody,
        config: PlanarPushingSimConfig,
    ) -> None:
        super().__init__()

        builder = DiagramBuilder()

        self.planar_pose_pub = builder.AddNamedSystem(
            "PlanarPoseTrajPublisher",
            PlanarPoseTrajPublisher(
                traj, config.mpc_config, config.delay_before_execution
            ),
        )

        self.pusher_pose_to_joint_pos = PusherPoseToJointPosDiffIk.add_to_builder(
            builder,
            time_step=config.time_step,
            use_diff_ik_feedback=False,
        )

        self.pusher_pose_controller = PusherPoseController.AddToBuilder(
            builder=builder,
            dynamics_config=config.dynamics_config,
            mpc_config=config.mpc_config,
            contact_mode_traj=self.planar_pose_pub.GetOutputPort("contact_mode_traj"),
            slider_planar_pose_traj=self.planar_pose_pub.GetOutputPort("slider_planar_pose_traj"),
            pusher_planar_pose_traj=self.planar_pose_pub.GetOutputPort("pusher_planar_pose_traj"),
            contact_force_traj=self.planar_pose_pub.GetOutputPort("contact_force_traj"),
            pose_cmd=self.pusher_pose_to_joint_pos.get_pose_input_port(),
            closed_loop=False,
            pusher_planar_pose_measured=None,
            slider_pose_measured=None,
        )

        # Export states
        builder.ExportOutput(
            self.pusher_pose_to_joint_pos.get_output_port(), "iiwa_position_cmd"
        )
        builder.BuildInto(self)
