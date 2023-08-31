import numpy as np
from pydrake.multibody.inverse_kinematics import (
    DifferentialInverseKinematicsIntegrator,
    DifferentialInverseKinematicsParameters,
)
from pydrake.multibody.parsing import (
    LoadModelDirectives,
    Parser,
    ProcessModelDirectives,
)
from pydrake.multibody.plant import MultibodyPlant
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pyparsing import Path

from planning_through_contact.geometry.planar.planar_pushing_trajectory import (
    PlanarPushingTrajectory,
)
from planning_through_contact.simulation.hardware.hardware_interface import (
    ManipulationHardwareInterface,
)
from planning_through_contact.simulation.planar_pushing.planar_pushing_diagram import (
    PlanarPushingSimConfig,
)
from planning_through_contact.simulation.planar_pushing.pusher_pose_to_joint_pos import (
    PusherPoseToJointPos,
)
from planning_through_contact.simulation.systems.pusher_pose_publisher import (
    PusherPosePublisher,
)


class PlanarPushingPositionControlNode:
    """
    An LCM node that implements an IK position controller for the end-effector of the Iiwa,
    following a pre-computed trajectory.
    """

    def __init__(
        self,
        traj: PlanarPushingTrajectory,
        delay_before_start: float = 10,
        config: PlanarPushingSimConfig = PlanarPushingSimConfig(),
    ):
        builder = DiagramBuilder()

        self.hardware_interface = builder.AddNamedSystem(
            "hardware_interface",
            ManipulationHardwareInterface(),
        )
        self.hardware_interface.Connect()

        self.pusher_pose_to_joint_pos = PusherPoseToJointPos.add_to_builder(
            builder,
            iiwa_joint_position_input=self.hardware_interface.GetInputPort(
                "iiwa_position"
            ),
            time_step=config.time_step,
        )

        self.pose_publisher = PusherPosePublisher.add_to_builder(
            builder,
            traj,
            delay_before_start,
            self.pusher_pose_to_joint_pos.get_input_port(),
        )

        self.diagram = builder.Build()

        self.simulator = Simulator(self.diagram)

        # This is important to avoid duplicate publishes to the hardware interface:
        self.simulator.set_publish_every_time_step(False)

        self.simulator.set_target_realtime_rate(1.0)

        self.hardware_interface_context = self.diagram.GetMutableSubsystemContext(
            self.hardware_interface, self.simulator.get_mutable_context()
        )

        # TODO(bernhardpg): Do we want to compute a feedforward torque?
        self.hardware_interface.GetInputPort("iiwa_feedforward_torque").FixValue(
            self.hardware_interface_context, np.zeros(7)
        )

        # If the diagram is only the hardware interface, then we must advance it a
        # little bit so that first LCM messages get processed. A simulated plant is
        # already publishing correct positions even without advancing, and indeed
        # we must not advance a simulated plant until the sliders and filters have
        # been initialized to match the plant.
        real_hardware = True  # TODO(bernhardpg)
        if real_hardware:
            self.simulator.AdvanceTo(1e-6)

        self._initialize_ik()

    def _initialize_ik(self) -> None:
        q0 = self.hardware_interface.GetOutputPort("iiwa_position_measured").Eval(
            self.hardware_interface_context
        )
        self.pusher_pose_to_joint_pos.init_diff_ik(
            q0, self.simulator.get_mutable_context()
        )

    def run(self, end_time: float = 1e8) -> None:
        self.simulator.AdvanceTo(end_time)

    def export_diagram(self, filename: str):
        import pydot

        gviz_string = self.diagram.GetGraphvizString()
        pydot.graph_from_dot_data(gviz_string)[0].write_png(filename)  # type: ignore
        print(f"Saved diagram to: {filename}")
