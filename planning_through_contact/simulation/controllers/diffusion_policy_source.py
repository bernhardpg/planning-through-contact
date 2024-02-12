
from pydrake.all import (
    DiagramBuilder,
    ZeroOrderHold,
)

from planning_through_contact.simulation.controllers.desired_planar_position_source_base import (
    DesiredPlanarPositionSourceBase,
)

from planning_through_contact.simulation.planar_pushing.planar_pushing_sim_config import (
    PlanarPushingSimConfig,
)

from planning_through_contact.simulation.planar_pushing.diffusion_policy_controller import (
    DiffusionPolicyController,
)

class DiffusionPolicySource(DesiredPlanarPositionSourceBase):
    """Uses the desired trajectory of the entire system and MPC controllers
    to generate desired positions for the robot."""

    def __init__(
        self,
        sim_config: PlanarPushingSimConfig,
        checkpoint: str
    ):
        super().__init__()

        self._sim_config = sim_config

        builder = DiagramBuilder()

        ## Add Leaf systems

        # Diffusion Policy Controller
        self._diffusion_policy_controller = builder.AddNamedSystem(
            "DiffusionPolicyController",
            DiffusionPolicyController(
                checkpoint,
                self._sim_config.pusher_start_pose,
                self._sim_config.slider_goal_pose,
                freq=10.0, # default
                delay=1.0, # default
            ),
        )

        # Zero Order Hold
        self._zero_order_hold = builder.AddNamedSystem(
            "ZeroOrderHold",
            ZeroOrderHold(1/200.0, vector_size=2),  # Just the x and y positions
        )


        ## Internal connections

        builder.Connect(
            self._diffusion_policy_controller.get_output_port(),
            self._zero_order_hold.get_input_port()
        )

        ## Export inputs and outputs (external)

        builder.ExportInput(
            self._diffusion_policy_controller.GetInputPort(
                "pusher_pose_measured"
            ),
            "pusher_pose_measured"
        )

        builder.ExportInput(
            self._diffusion_policy_controller.GetInputPort("camera"),
            "camera"
        )

        builder.ExportOutput(
            self._zero_order_hold.get_output_port(),
            "planar_position_command"
        )

        builder.BuildInto(self)