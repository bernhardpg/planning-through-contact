
from pydrake.all import (
    DiagramBuilder,
    OutputPort,
    Diagram,
)
from planning_through_contact.geometry.planar.planar_pushing_trajectory import PlanarPushingTrajectory

from planning_through_contact.simulation.controllers.desired_position_source_base import DesiredPositionSourceBase
from planning_through_contact.simulation.planar_pushing.planar_pose_traj_publisher import PlanarPoseTrajPublisher
from planning_through_contact.simulation.planar_pushing.planar_pushing_diagram import PlanarPushingSimConfig
from planning_through_contact.simulation.planar_pushing.pusher_pose_controller import PusherPoseController


class MPCPositionSource(DesiredPositionSourceBase):
    """Uses the desired trajectory of the entire system and MPC controllers
    to generate desired positions for the robot."""

    def __init__(
        self,
        sim_config: PlanarPushingSimConfig,
        traj: PlanarPushingTrajectory,
    ):
        self._sim_config = sim_config
        self._traj = traj

    def setup(self, builder: DiagramBuilder, state_estimator: Diagram) -> OutputPort:
        """Setup the desired position source (MPC)."""

        # Desired trajectory sources for pusher and slider
        self.planar_pose_pub = builder.AddNamedSystem(
            "PlanarPoseTrajPublisher",
            PlanarPoseTrajPublisher(
                self._traj, self._sim_config.mpc_config, self._sim_config.delay_before_execution
            ),
        )

        # MPC controllers
        self.pusher_pose_controller = PusherPoseController.AddToBuilder(
            builder=builder,
            dynamics_config=self._sim_config.dynamics_config,
            mpc_config=self._sim_config.mpc_config,
            contact_mode_traj=self.planar_pose_pub.GetOutputPort("contact_mode_traj"),
            slider_planar_pose_traj=self.planar_pose_pub.GetOutputPort("slider_planar_pose_traj"),
            pusher_planar_pose_traj=self.planar_pose_pub.GetOutputPort("pusher_planar_pose_traj"),
            contact_force_traj=self.planar_pose_pub.GetOutputPort("contact_force_traj"),
            pose_cmd=None, # Connect this in environment
            closed_loop=self._sim_config.closed_loop,
            pusher_planar_pose_measured=state_estimator.GetOutputPort("pusher_pose"),
            slider_pose_measured=state_estimator.GetOutputPort("slider_pose"),
        )
        # Last system of the PusherPoseController, it's output is not connected
        zero_order_hold = builder.GetSubsystemByName("ZeroOrderHold")

        return zero_order_hold.get_output_port()
