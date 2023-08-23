import pygame
from drake.lcmt_iiwa_command import lcmt_iiwa_command
from drake.lcmt_iiwa_status import lcmt_iiwa_status
from pydrake.common.value import AbstractValue
from pydrake.lcm import DrakeLcm
from pydrake.manipulation import IiwaCommandSender, IiwaStatusReceiver
from pydrake.systems.framework import Diagram, DiagramBuilder
from pydrake.systems.lcm import (
    LcmInterfaceSystem,
    LcmPublisherSystem,
    LcmSubscriberSystem,
)
from pygame.locals import *


class ManipulationHardwareInterface(Diagram):
    def __init__(self):
        Diagram.__init__(self)

        builder = DiagramBuilder()
        self.lcm = DrakeLcm()
        lcm_system = builder.AddSystem(LcmInterfaceSystem(self.lcm))

        # Publish iiwa command
        iiwa_command_publisher = builder.AddSystem(
            LcmPublisherSystem.Make(
                channel="IIWA_COMMAND",
                lcm_type=lcmt_iiwa_command,
                lcm=self.lcm,
                publish_period=0.005,
                use_cpp_serializer=True,
            )
        )
        iiwa_command_sender = builder.AddSystem(IiwaCommandSender())
        builder.ExportInput(
            iiwa_command_sender.get_position_input_port(), "iiwa_position"
        )
        builder.ExportInput(
            iiwa_command_sender.get_torque_input_port(), "iiwa_feedforward_torque"
        )
        builder.Connect(
            iiwa_command_sender.get_output_port(),
            iiwa_command_publisher.get_input_port(),
        )

        # Subscribe to iiwa status
        self.iiwa_status_subscriber = builder.AddSystem(
            LcmSubscriberSystem.Make(
                channel="IIWA_STATUS",
                lcm_type=lcmt_iiwa_status,
                lcm=self.lcm,
                use_cpp_serializer=True,
            )
        )
        iiwa_status_receiver = builder.AddSystem(IiwaStatusReceiver())
        builder.Connect(
            self.iiwa_status_subscriber.get_output_port(),
            iiwa_status_receiver.get_input_port(),
        )
        builder.ExportOutput(
            iiwa_status_receiver.get_position_commanded_output_port(),
            "iiwa_position_commanded",
        )
        builder.ExportOutput(
            iiwa_status_receiver.get_position_measured_output_port(),
            "iiwa_position_measured",
        )
        builder.ExportOutput(
            iiwa_status_receiver.get_velocity_estimated_output_port(),
            "iiwa_velocity_estimated",
        )
        builder.ExportOutput(
            iiwa_status_receiver.get_torque_commanded_output_port(),
            "iiwa_torque_commanded",
        )
        builder.ExportOutput(
            iiwa_status_receiver.get_torque_measured_output_port(),
            "iiwa_torque_measured",
        )
        builder.ExportOutput(
            iiwa_status_receiver.get_torque_external_output_port(),
            "iiwa_torque_external",
        )

        builder.BuildInto(self)
        self.set_name("manipulation_hardware_interface")

    def Connect(self):
        for attempt in range(10):
            value = AbstractValue.Make(lcmt_iiwa_status())
            count = self.iiwa_status_subscriber.WaitForMessage(10, value, timeout=0.1)
            self.lcm.HandleSubscriptions(0)

        if count < 9:  # type: ignore
            raise RuntimeError("couldn't connect to lcm.")
