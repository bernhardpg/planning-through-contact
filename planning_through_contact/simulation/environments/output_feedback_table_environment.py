import logging
import os
from typing import Optional

import numpy as np
from pydrake.all import (
    ConstantVectorSource,
    Demultiplexer,
    DiagramBuilder,
    LogVectorOutput,
    Meshcat,
    Simulator,
)

from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.simulation.controllers.desired_planar_position_source_base import (
    DesiredPlanarPositionSourceBase,
)
from planning_through_contact.simulation.controllers.cylinder_actuated_station import (
    CylinderActuatedStation
)
from planning_through_contact.simulation.controllers.iiwa_hardware_station import (
    IiwaHardwareStation,
)
from planning_through_contact.simulation.controllers.robot_system_base import (
    RobotSystemBase,
)
from planning_through_contact.simulation.controllers.teleop_position_source import (
    TeleopPositionSource,
)
from planning_through_contact.simulation.planar_pushing.planar_pushing_sim_config import (
    PlanarPushingSimConfig,
)
from planning_through_contact.simulation.sensors.optitrack_config import OptitrackConfig

from planning_through_contact.simulation.systems.rigid_transform_to_planar_pose_vector_system import (
    RigidTransformToPlanarPoseVectorSystem,
)
from planning_through_contact.simulation.systems.robot_state_to_rigid_transform import (
    RobotStateToRigidTransform,
)
from planning_through_contact.visualize.analysis import (
    plot_joint_state_logs,
    plot_and_save_planar_pushing_logs_from_sim,
)

logger = logging.getLogger(__name__)


class OutputFeedbackTableEnvironment:
    def __init__(
        self,
        desired_position_source: DesiredPlanarPositionSourceBase,
        robot_system: RobotSystemBase,
        sim_config: PlanarPushingSimConfig,
        station_meshcat: Optional[Meshcat] = None,
    ):
        self._desired_position_source = desired_position_source
        self._robot_system = robot_system
        self._sim_config = sim_config
        self._meshcat = station_meshcat
        self._simulator = None

        self._plant = self._robot_system.station_plant
        self._slider = self._robot_system.slider

        builder = DiagramBuilder()

        ## Add systems

        builder.AddNamedSystem(
            "DesiredPlanarPositionSource",
            self._desired_position_source,
        )

        builder.AddNamedSystem(
            "PositionController",
            self._robot_system,
        )

        # TODO: hacky way to get z value. Works for box and tee
        if self._sim_config.slider.name == "box":
            z_value = self._sim_config.slider.geometry.height / 2.0
        else: # T
            z_value = self._sim_config.slider.geometry.box_1.height / 2.0

        self._robot_state_to_rigid_transform = builder.AddNamedSystem(
            "RobotStateToRigidTransform",
            RobotStateToRigidTransform(
                self._robot_system.station_plant, 
                self._robot_system.robot_model_name,
                z_value=z_value
            ),
        )

        self._meshcat = self._robot_system._meshcat


        ## Connect systems

        # Connect PositionController to RobotStateToOutputs
        builder.Connect(
            self._robot_system.GetOutputPort("robot_state_measured"),
            self._robot_state_to_rigid_transform.GetInputPort("robot_state"),
        )

        # Inputs to desired position source
        builder.Connect(
            self._robot_state_to_rigid_transform.GetOutputPort("pose"),
            self._desired_position_source.GetInputPort("pusher_pose_measured"),
        )
        builder.Connect(
            self._robot_system.GetOutputPort("rgbd_sensor_overhead_camera"),
            self._desired_position_source.GetInputPort("camera"),
        )

        # Inputs to robot system
        builder.Connect(
            self._desired_position_source.GetOutputPort("planar_position_command"),
            self._robot_system.GetInputPort("planar_position_command"),
        )

        diagram = builder.Build()
        self._diagram = diagram

        self._simulator = Simulator(diagram)
        if sim_config.use_realtime:
            self._simulator.set_target_realtime_rate(1.0)

        self.context = self._simulator.get_mutable_context()
        self._robot_system.pre_sim_callback(self.context)

        # initialize slider above the table
        self.mbp_context = self._plant.GetMyContextFromRoot(self.context)
        self.set_slider_planar_pose(self._sim_config.slider_start_pose)

    def export_diagram(self, filename: str):
        import pydot

        pydot.graph_from_dot_data(self._diagram.GetGraphvizString())[0].write_pdf(  # type: ignore
            filename
        )
        print(f"Saved diagram to: {filename}")

    def set_slider_planar_pose(self, pose: PlanarPose):
        min_height = 0.05

        # add a small height to avoid the box penetrating the table
        q = pose.to_generalized_coords(min_height + 1e-2, z_axis_is_positive=True)
        self._plant.SetPositions(self.mbp_context, self._slider, q)

    def simulate(
        self,
        timeout=1e8,
        recording_file: Optional[str] = None,
        save_dir: str = "",
        for_reset: bool = False,
    ) -> None:
        """
        :return: Returns a tuple of (success, simulation_time_s).
        """
        if recording_file:
            self._meshcat.StartRecording()
        time_step = self._sim_config.time_step * 10
        if not isinstance(self._desired_position_source, TeleopPositionSource):
            for t in np.append(np.arange(0, timeout, time_step), timeout):
                self._simulator.AdvanceTo(t)
                # self._visualize_desired_slider_pose(t)
                # Print the time every 5 seconds
                if t % 5 == 0:
                    logger.info(f"t={t}")

        else:
            self._simulator.AdvanceTo(timeout)

        self.save_logs(recording_file, save_dir)

    # TODO: write function to visualize target pose
    # def _visualize_desired_slider_pose(self, t):
    #     # Visualizing the desired slider pose
    #     context = self._desired_position_source.GetMyContextFromRoot(self.context)
    #     slider_desired_pose_vec = self._desired_position_source.GetOutputPort(
    #         "desired_slider_planar_pose_vector"
    #     ).Eval(context)
    #     self._state_estimator._visualize_desired_slider_pose(
    #         PlanarPose(*slider_desired_pose_vec),
    #         time_in_recording=t,
    #     )
    #     pusher_desired_pose_vec = self._desired_position_source.GetOutputPort(
    #         "planar_position_command"
    #     ).Eval(context)
    #     self._state_estimator._visualize_desired_pusher_pose(
    #         PlanarPose(*pusher_desired_pose_vec, 0),
    #         time_in_recording=t,
    #     )