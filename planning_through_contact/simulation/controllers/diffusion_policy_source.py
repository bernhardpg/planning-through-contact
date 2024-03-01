import numpy as np

from pydrake.all import (
    DiagramBuilder,
    ZeroOrderHold,
    LeafSystem,
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
        checkpoint: str,
        diffusion_policy_path: str = "/home/adam/workspace/gcs-diffusion",
    ):
        super().__init__()

        self._sim_config = sim_config

        builder = DiagramBuilder()

        ## Add Leaf systems

        # Diffusion Policy Controller
        freq = 10.0
        self._diffusion_policy_controller = builder.AddNamedSystem(
            "DiffusionPolicyController",
            DiffusionPolicyController(
                checkpoint=checkpoint,
                diffusion_policy_path=diffusion_policy_path,
                initial_pusher_pose=self._sim_config.pusher_start_pose,
                target_slider_pose=self._sim_config.slider_goal_pose,
                freq=freq,
                delay=1.0, # default
            ),
        )

        # Zero Order Hold

        self._zero_order_hold = builder.AddNamedSystem(
            "ZeroOrderHold",
            ZeroOrderHold(
                period_sec=1/freq, 
                vector_size=2, 
                # offset_sec=1/(2.0*freq)
            ),  # Just the x and y positions
        )

        # AppendZeros
        self._append_zeros = builder.AddSystem(
            AppendZeros(input_size=2, num_zeros=1)
        )


        ## Internal connections

        builder.Connect(
            self._diffusion_policy_controller.get_output_port(),
            self._zero_order_hold.get_input_port()
        )

        builder.Connect(
            self._zero_order_hold.get_output_port(),
            self._append_zeros.get_input_port()
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

        builder.ExportOutput(
            self._append_zeros.get_output_port(),
            "planar_pose_command"
        )

        builder.BuildInto(self)


class AppendZeros(LeafSystem):
    def __init__(self, input_size:int, num_zeros: int):
        super().__init__()
        self._input_size = input_size
        self._num_zeros = num_zeros
        self.DeclareVectorInputPort("input", input_size)
        self.DeclareVectorOutputPort("output", input_size+num_zeros, self.CalcOutput)

    def CalcOutput(self, context, output):
        input = self.EvalVectorInput(context, 0).get_value()
        output_vec = np.zeros(self._input_size + self._num_zeros)
        output_vec[:self._input_size] = input
        output.SetFromVector(output_vec)