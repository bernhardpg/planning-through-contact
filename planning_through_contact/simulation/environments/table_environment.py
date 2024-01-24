import logging
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
from planning_through_contact.simulation.state_estimators.state_estimator import (
    StateEstimator,
)
from planning_through_contact.simulation.systems.rigid_transform_to_planar_pose_vector_system import (
    RigidTransformToPlanarPoseVectorSystem,
)
from planning_through_contact.visualize.analysis import (
    plot_joint_state_logs,
    plot_planar_pushing_logs_from_pose_vectors,
)

logger = logging.getLogger(__name__)


class TableEnvironment:
    def __init__(
        self,
        desired_position_source: DesiredPlanarPositionSourceBase,
        robot_system: RobotSystemBase,
        sim_config: PlanarPushingSimConfig,
        optitrack_config: OptitrackConfig,
        station_meshcat: Optional[Meshcat] = None,
        state_estimator_meshcat: Optional[Meshcat] = None,
    ):
        self._desired_position_source = desired_position_source
        self._robot_system = robot_system
        self._sim_config = sim_config
        self._meshcat = station_meshcat
        self._simulator = None

        builder = DiagramBuilder()

        ## Add systems

        self._state_estimator = builder.AddNamedSystem(
            "state_estimator",
            StateEstimator(
                sim_config=sim_config,
                meshcat=state_estimator_meshcat,
                robot_model_name=robot_system.robot_model_name,
            ),
        )

        builder.AddNamedSystem(
            "DesiredPlanarPositionSource",
            self._desired_position_source,
        )

        builder.AddNamedSystem(
            "PositionController",
            self._robot_system,
        )
        self._meshcat = self._robot_system._meshcat

        if sim_config.use_hardware:
            from planning_through_contact.simulation.sensors.optitrack import (
                OptitrackObjectTransformUpdaterDiagram,
            )

            optitrack_object_transform_updater: OptitrackObjectTransformUpdaterDiagram = builder.AddNamedSystem(
                "OptitrackTransformUpdater",
                OptitrackObjectTransformUpdaterDiagram(
                    state_estimator=self._state_estimator,
                    optitrack_iiwa_id=optitrack_config.iiwa_id,
                    optitrack_body_id=optitrack_config.slider_id,
                    X_optitrackBody_plantBody=optitrack_config.X_optitrackBody_plantBody,
                ),
            )

        ## Connect systems

        # Inputs to desired position source
        if self._sim_config.closed_loop:
            builder.Connect(
                self._state_estimator.GetOutputPort("query_object"),
                self._desired_position_source.GetInputPort("query_object"),
            )
            builder.Connect(
                self._state_estimator.GetOutputPort("slider_pose_estimated"),
                self._desired_position_source.GetInputPort("slider_pose_estimated"),
            )
            builder.Connect(
                self._state_estimator.GetOutputPort("pusher_pose_estimated"),
                self._desired_position_source.GetInputPort("pusher_pose_estimated"),
            )

        # Inputs to robot system
        builder.Connect(
            self._desired_position_source.GetOutputPort("planar_position_command"),
            self._robot_system.GetInputPort("planar_position_command"),
        )

        # Inputs to state estimator
        # Connections to update the robot state within state estimator
        builder.Connect(
            self._robot_system.GetOutputPort("robot_state_measured"),
            self._state_estimator.GetInputPort("robot_state"),
        )
        if sim_config.use_hardware:
            # TODO connect Optitrack system
            # For now set the object_position to be constant
            # height = 0.025
            # q_slider = sim_config.slider_start_pose.to_generalized_coords(
            #     height, z_axis_is_positive=True
            # )
            # const_object_position = builder.AddSystem(ConstantVectorSource(q_slider))
            # builder.Connect(
            #     const_object_position.get_output_port(),
            #     self._state_estimator.GetInputPort("object_position"),
            # )
            pass
        else:
            # Connections to update the object position within state estimator
            self._plant = self._robot_system.station_plant
            self._slider = self._robot_system.slider
            slider_demux = builder.AddSystem(
                Demultiplexer(
                    [
                        self._plant.num_positions(self._slider),
                        self._plant.num_velocities(self._slider),
                    ]
                )
            )
            builder.Connect(
                self._robot_system.GetOutputPort("object_state_measured"),
                slider_demux.get_input_port(),
            )
            builder.Connect(
                slider_demux.get_output_port(0),
                self._state_estimator.GetInputPort("object_position"),
            )

        # Will break if save plots during teleop
        if sim_config.save_plots:
            assert not isinstance(
                self._desired_position_source, TeleopPositionSource
            ), "Cannot save plots during teleop"
            # Actual State Loggers
            pusher_pose_to_vector = builder.AddSystem(
                RigidTransformToPlanarPoseVectorSystem()
            )
            builder.Connect(
                self._state_estimator.GetOutputPort("pusher_pose_estimated"),
                pusher_pose_to_vector.get_input_port(),
            )
            pusher_pose_logger = LogVectorOutput(
                pusher_pose_to_vector.get_output_port(), builder
            )
            slider_pose_to_vector = builder.AddSystem(
                RigidTransformToPlanarPoseVectorSystem()
            )
            builder.Connect(
                self._state_estimator.GetOutputPort("slider_pose_estimated"),
                slider_pose_to_vector.get_input_port(),
            )
            slider_pose_logger = LogVectorOutput(
                slider_pose_to_vector.get_output_port(), builder
            )
            # Desired State Loggers
            # "desired_pusher_planar_pose_vector" is the reference trajectory
            # "planar_position_command" is the commanded trajectory (e.g. after passing through MPC if closed loop or just the reference trajectory if open loop)
            pusher_pose_desired_logger = LogVectorOutput(
                self._desired_position_source.GetOutputPort("planar_position_command"),
                builder,
            )
            slider_pose_desired_logger = LogVectorOutput(
                self._desired_position_source.GetOutputPort(
                    "desired_slider_planar_pose_vector"
                ),
                builder,
            )
            self._joint_state_logger = LogVectorOutput(
                self._robot_system.GetOutputPort("robot_state_measured"), builder
            )

            self._pusher_pose_logger = pusher_pose_logger
            self._slider_pose_logger = slider_pose_logger
            self._pusher_pose_desired_logger = pusher_pose_desired_logger
            self._slider_pose_desired_logger = slider_pose_desired_logger

        diagram = builder.Build()
        self._diagram = diagram

        self._simulator = Simulator(diagram)
        if sim_config.use_realtime:
            self._simulator.set_target_realtime_rate(1.0)

        self.context = self._simulator.get_mutable_context()
        self._robot_system.pre_sim_callback(self.context)
        if sim_config.use_hardware:
            optitrack_object_transform_updater.set_plant_context(
                self._state_estimator.get_plant_context()
            )
        else:
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

    def simulate(self, timeout=1e8, save_recording_as: Optional[str] = None) -> None:
        """
        :return: Returns a tuple of (success, simulation_time_s).
        """
        if save_recording_as:
            self._state_estimator.meshcat.StartRecording()
            self._meshcat.StartRecording()
        time_step = self._sim_config.time_step * 10
        if not isinstance(self._desired_position_source, TeleopPositionSource):
            for t in np.append(np.arange(0, timeout, time_step), timeout):
                self._simulator.AdvanceTo(t)
                # Visualizing the desired slider pose
                context = self._desired_position_source.GetMyContextFromRoot(
                    self.context
                )
                slider_desired_pose_vec = self._desired_position_source.GetOutputPort(
                    "desired_slider_planar_pose_vector"
                ).Eval(context)
                self._state_estimator._visualize_desired_slider_pose(
                    PlanarPose(*slider_desired_pose_vec),
                    time_in_recording=t,
                )
                pusher_desired_pose_vec = self._desired_position_source.GetOutputPort(
                    "planar_position_command"
                ).Eval(context)
                self._state_estimator._visualize_desired_pusher_pose(
                    PlanarPose(*pusher_desired_pose_vec, 0),
                    time_in_recording=t,
                )
                # Print the time every 5 seconds
                if t % 5 == 0:
                    logger.info(f"t={t}")

        else:
            self._simulator.AdvanceTo(timeout)
        if save_recording_as:
            self._meshcat.StopRecording()
            self._meshcat.SetProperty("/drake/contact_forces", "visible", False)
            self._meshcat.PublishRecording()
            self._state_estimator.meshcat.StopRecording()
            self._state_estimator.meshcat.SetProperty(
                "/drake/contact_forces", "visible", True
            )
            self._state_estimator.meshcat.PublishRecording()
            res = self._state_estimator.meshcat.StaticHtml()
            with open(save_recording_as, "w") as f:
                f.write(res)
        if self._sim_config.save_plots:
            pusher_pose_log = self._pusher_pose_logger.FindLog(self.context)
            slider_pose_log = self._slider_pose_logger.FindLog(self.context)
            pusher_pose_desired_log = self._pusher_pose_desired_logger.FindLog(
                self.context
            )
            slider_pose_desired_log = self._slider_pose_desired_logger.FindLog(
                self.context
            )
            plot_planar_pushing_logs_from_pose_vectors(
                pusher_pose_log,
                slider_pose_log,
                pusher_pose_desired_log,
                slider_pose_desired_log,
            )
            plot_joint_state_logs(
                self._joint_state_logger.FindLog(self.context),
                self._robot_system.robot.num_positions(),
            )
