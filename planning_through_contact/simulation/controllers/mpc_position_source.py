from pydrake.all import DiagramBuilder, ZeroOrderHold

from planning_through_contact.geometry.planar.planar_pushing_trajectory import (
    PlanarPushingTrajectory,
)
from planning_through_contact.simulation.controllers.desired_planar_position_source_base import (
    DesiredPlanarPositionSourceBase,
)
from planning_through_contact.simulation.planar_pushing.planar_pose_traj_publisher import (
    PlanarPoseTrajPublisher,
)
from planning_through_contact.simulation.planar_pushing.planar_pushing_sim_config import (
    PlanarPushingSimConfig,
)
from planning_through_contact.simulation.planar_pushing.pusher_pose_controller import (
    PusherPoseController,
)
from planning_through_contact.simulation.systems.contact_detection_system import (
    ContactDetectionSystem,
)


class MPCPositionSource(DesiredPlanarPositionSourceBase):
    """Uses the desired trajectory of the entire system and MPC controllers
    to generate desired positions for the robot."""

    def __init__(
        self,
        sim_config: PlanarPushingSimConfig,
        traj: PlanarPushingTrajectory,
    ):
        super().__init__()

        self._sim_config = sim_config
        self._traj = traj

        builder = DiagramBuilder()

        ## Add Leaf systems

        # Desired trajectory sources for pusher and slider
        self._planar_pose_pub = builder.AddNamedSystem(
            "PlanarPoseTrajPublisher",
            PlanarPoseTrajPublisher(
                self._traj,
                self._sim_config.mpc_config,
                self._sim_config.delay_before_execution,
            ),
        )

        # Contact Detection System that tells the pusher pose controller when the contact is broken between pusher and slider
        self._contact_detector = builder.AddNamedSystem(
            "ContactDetectionSystem",
            ContactDetectionSystem(
                "pusher::collision",
                self._sim_config.slider.geometry.collision_geometry_names,
            ),
        )

        # MPC controller
        self._pusher_pose_controller = builder.AddNamedSystem(
            "PusherPoseController",
            PusherPoseController(
                dynamics_config=self._sim_config.dynamics_config,
                mpc_config=self._sim_config.mpc_config,
                closed_loop=self._sim_config.closed_loop,
            ),
        )

        # Zero order hold for desired planar pose
        period = 1 / self._sim_config.mpc_config.rate_Hz
        self._zero_order_hold = builder.AddNamedSystem(
            "ZeroOrderHold",
            ZeroOrderHold(period, vector_size=2),  # Just the x and y positions
        )

        ## Internal connections

        builder.Connect(
            self._contact_detector.GetOutputPort("contact_detected"),
            self._pusher_pose_controller.GetInputPort("pusher_slider_contact"),
        )

        builder.Connect(
            self._pusher_pose_controller.GetOutputPort("translation"),
            self._zero_order_hold.get_input_port(),
        )

        builder.Connect(
            self._planar_pose_pub.GetOutputPort("contact_mode_traj"),
            self._pusher_pose_controller.GetInputPort("contact_mode_traj"),
        )
        builder.Connect(
            self._planar_pose_pub.GetOutputPort("pusher_planar_pose_traj"),
            self._pusher_pose_controller.GetInputPort("pusher_planar_pose_traj"),
        )
        builder.Connect(
            self._planar_pose_pub.GetOutputPort("slider_planar_pose_traj"),
            self._pusher_pose_controller.GetInputPort("slider_planar_pose_traj"),
        )
        builder.Connect(
            self._planar_pose_pub.GetOutputPort("contact_force_traj"),
            self._pusher_pose_controller.GetInputPort("contact_force_traj"),
        )

        ## Export inputs and outputs (external)

        builder.ExportInput(
            self._contact_detector.GetInputPort("query_object"), "query_object"
        )

        builder.ExportInput(
            self._pusher_pose_controller.GetInputPort("slider_pose_estimated"),
            "slider_pose_estimated",
        )

        builder.ExportInput(
            self._pusher_pose_controller.GetInputPort("pusher_pose_estimated"),
            "pusher_pose_estimated",
        )

        builder.ExportOutput(
            self._zero_order_hold.get_output_port(), "planar_position_command"
        )

        # Reference trajectory for pusher
        builder.ExportOutput(
            self._planar_pose_pub.GetOutputPort("desired_pusher_planar_pose_vector"),
            "desired_pusher_planar_pose_vector",
        )

        # Reference trajectory for slider
        builder.ExportOutput(
            self._planar_pose_pub.GetOutputPort("desired_slider_planar_pose_vector"),
            "desired_slider_planar_pose_vector",
        )

        # For logging only
        builder.ExportOutput(
            self._pusher_pose_controller.GetOutputPort("mpc_control"),
            "mpc_control",
        )
        builder.ExportOutput(
            self._pusher_pose_controller.GetOutputPort("mpc_control_desired"),
            "mpc_control_desired",
        )

        builder.BuildInto(self)
