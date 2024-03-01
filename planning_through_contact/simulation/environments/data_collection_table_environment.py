import logging
import os
from typing import Optional

import numpy as np
import pickle
from pydrake.all import (
    ConstantVectorSource,
    Demultiplexer,
    DiagramBuilder,
    LogVectorOutput,
    Meshcat,
    Simulator,
    StateInterpolatorWithDiscreteDerivative
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
from planning_through_contact.simulation.state_estimators.state_estimator import (
    StateEstimator,
)
from planning_through_contact.simulation.systems.rigid_transform_to_planar_pose_vector_system import (
    RigidTransformToPlanarPoseVectorSystem,
)
from planning_through_contact.simulation.systems.planar_pose_to_generalized_coords import (
    PlanarPoseToGeneralizedCoords,
)
from planning_through_contact.visualize.analysis import (
    plot_joint_state_logs,
    plot_and_save_planar_pushing_logs_from_sim,
    PlanarPushingLog,
    CombinedPlanarPushingLogs
)

logger = logging.getLogger(__name__)


class DataCollectionTableEnvironment:
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
        
        # Add system to convert desired position to desired position and velocity.
        self._desired_state_source = builder.AddNamedSystem(
            "DesiredStateInterpolator",
            StateInterpolatorWithDiscreteDerivative(
                self._robot_system._num_positions,
                self._sim_config.time_step,
                suppress_initial_transient=True,
            ),
        )

        # Add system to convert slider_pose to generalized coords
        # TODO: better way to get z_value
        z_value = 0.025
        self._slider_pose_to_generalized_coords = builder.AddNamedSystem(
            "PlanarPoseToGeneralizedCoords",
            PlanarPoseToGeneralizedCoords(
                z_value=z_value,
                z_axis_is_positive=True,
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
            self._desired_position_source.GetOutputPort("planar_position_command"),
            self._desired_state_source.get_input_port(),
        )

        builder.Connect(
            self._desired_state_source.get_output_port(),
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
            self._plant = self._robot_system.station_plant
            self._slider = self._robot_system.slider
            
            # Connections to update the object position within state estimator
            builder.Connect(
                self._desired_position_source.GetOutputPort(
                    "desired_slider_planar_pose_vector"
                ),
                self._slider_pose_to_generalized_coords.get_input_port()
            )
            builder.Connect(
                self._slider_pose_to_generalized_coords.get_output_port(),
                self._state_estimator.GetInputPort("object_position"),
            )

        # Will break if save plots during teleop
        if sim_config.save_plots or sim_config.collect_data:
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
            self._pusher_pose_logger = LogVectorOutput(
                pusher_pose_to_vector.get_output_port(), builder
            )
            slider_pose_to_vector = builder.AddSystem(
                RigidTransformToPlanarPoseVectorSystem()
            )
            builder.Connect(
                self._state_estimator.GetOutputPort("slider_pose_estimated"),
                slider_pose_to_vector.get_input_port(),
            )
            self._slider_pose_logger = LogVectorOutput(
                slider_pose_to_vector.get_output_port(), builder
            )

            # Desired State Loggers
            # "desired_pusher_planar_pose_vector" is the reference trajectory
            # "planar_position_command" is the commanded trajectory (e.g. after passing through MPC if closed loop or just the reference trajectory if open loop)
            self._pusher_pose_desired_logger = LogVectorOutput(
                self._desired_position_source.GetOutputPort("planar_position_command"),
                builder,
            )
            self._slider_pose_desired_logger = LogVectorOutput(
                self._desired_position_source.GetOutputPort(
                    "desired_slider_planar_pose_vector"
                ),
                builder,
            )
            self._joint_state_logger = LogVectorOutput(
                self._robot_system.GetOutputPort("robot_state_measured"), builder
            )
            # Actual command logger
            self._control_logger = LogVectorOutput(
                self._desired_position_source.GetOutputPort("mpc_control"), builder
            )
            # Desired command logger
            self._control_desired_logger = LogVectorOutput(
                self._desired_position_source.GetOutputPort("mpc_control_desired"),
                builder,
            )
        
        if sim_config.collect_data:
            assert sim_config.camera_config is not None
            from pydrake.systems.sensors import (
                ImageWriter,
                PixelType
            )

            if not os.path.exists(sim_config.data_dir):
                os.mkdir(sim_config.data_dir)
            traj_idx = 0
            for path in os.listdir(sim_config.data_dir):
                if os.path.isdir(os.path.join(sim_config.data_dir, path)):
                    traj_idx += 1
            image_dir = f'{sim_config.data_dir}/{traj_idx}/images'
            os.makedirs(image_dir)
                                    
            image_writer_system = ImageWriter()
            image_writer_system.DeclareImageInputPort(
                pixel_type=PixelType.kRgba8U,
                port_name="overhead_camera_image",
                file_name_format= image_dir + '/{time_msec}.png',
                publish_period=0.1,
                start_time=0.0
            )
            image_writer = builder.AddNamedSystem(
                "ImageWriter",
                image_writer_system
            )
            builder.Connect(
                self._state_estimator.GetOutputPort(
                    "rgbd_sensor_state_estimator_overhead_camera"
                ),
                image_writer.get_input_port()
            )

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
            self._state_estimator.meshcat.StartRecording()
            self._meshcat.StartRecording()
        time_step = self._sim_config.time_step * 10
        if for_reset:
            self._state_estimator.meshcat.AddButton("Reset Done!")
            t = 0
            while (
                self._state_estimator.meshcat.GetButtonClicks("Reset Done!") < 1
                and t < timeout
            ):
                t += time_step
                self._simulator.AdvanceTo(t)
                self._visualize_desired_slider_pose(t)
            self._state_estimator.meshcat.DeleteAddedControls()
            return
        if not isinstance(self._desired_position_source, TeleopPositionSource):
            for t in np.append(np.arange(0, timeout, time_step), timeout):
                self._simulator.AdvanceTo(t)
                self._visualize_desired_slider_pose(t)
                # Print the time every 5 seconds
                if t % 5 == 0:
                    logger.info(f"t={t}")

        else:
            self._simulator.AdvanceTo(timeout)
        self.save_logs(recording_file, save_dir)
        self.save_data()
        
    
    def save_logs(self, recording_file: Optional[str], save_dir: str):
        if recording_file:
            self._meshcat.StopRecording()
            self._meshcat.SetProperty("/drake/contact_forces", "visible", False)
            self._meshcat.PublishRecording()
            self._state_estimator.meshcat.StopRecording()
            self._state_estimator.meshcat.SetProperty(
                "/drake/contact_forces", "visible", True
            )
            self._state_estimator.meshcat.PublishRecording()
            res = self._state_estimator.meshcat.StaticHtml()
            if save_dir:
                recording_file = os.path.join(save_dir, recording_file)
            with open(recording_file, "w") as f:
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
            control_log = self._control_logger.FindLog(self.context)
            control_desired_log = self._control_desired_logger.FindLog(self.context)

            last_planar_pose = slider_pose_log.data()[:, -1]
            last_desired_planar_pose = slider_pose_desired_log.data()[:, -1]
            error = last_planar_pose - last_desired_planar_pose
            error_norm = np.linalg.norm(error)
            logger.info(
                f"\nSUMMARY:\n last planar pose: {last_planar_pose}"
                f"\n last desired planar pose: {last_desired_planar_pose}"
                f"\n error: {error}\n error norm: {error_norm:.4f}"
            )
            plot_and_save_planar_pushing_logs_from_sim(
                pusher_pose_log,
                slider_pose_log,
                control_log,
                control_desired_log,
                pusher_pose_desired_log,
                slider_pose_desired_log,
                save_dir=save_dir,
            )

            # Hack to get num_positions from cylinder object
            num_positions = None
            if isinstance(self._robot_system, CylinderActuatedStation):
                num_positions = self._robot_system._num_positions
            else:
                num_positions = self._robot_system.robot.num_positions()
            plot_joint_state_logs(
                self._joint_state_logger.FindLog(self.context),
                num_positions,
                save_dir=save_dir,
            )
    
    def save_data(self):
        import pathlib

        if self._sim_config.collect_data:
            assert self._sim_config.data_dir is not None

            # Save the logs
            pusher_pose_log = self._pusher_pose_logger.FindLog(self.context)
            slider_pose_log = self._slider_pose_logger.FindLog(self.context)
            pusher_pose_desired_log = self._pusher_pose_desired_logger.FindLog(
                self.context
            )
            slider_pose_desired_log = self._slider_pose_desired_logger.FindLog(
                self.context
            )
            control_log = self._control_logger.FindLog(self.context)
            control_desired_log = self._control_desired_logger.FindLog(self.context)

            pusher_actual = PlanarPushingLog.from_pose_vector_log(pusher_pose_log)
            slider_actual = PlanarPushingLog.from_log(slider_pose_log, control_log)
            pusher_desired = PlanarPushingLog.from_pose_vector_log(
                pusher_pose_desired_log
            )
            slider_desired = PlanarPushingLog.from_log(
                slider_pose_desired_log,
                control_desired_log,
            )
            combined = CombinedPlanarPushingLogs(
                pusher_actual=pusher_actual,
                slider_actual=slider_actual,
                pusher_desired=pusher_desired,
                slider_desired=slider_desired,
            )

            # assumes that a directory for this trajectory has already been
            # created (when saving the images)
            traj_idx = -1
            for path in os.listdir(self._sim_config.data_dir):
                if os.path.isdir(os.path.join(self._sim_config.data_dir, path)):
                    traj_idx += 1
            assert traj_idx >= 0
            save_dir = pathlib.Path(self._sim_config.data_dir).joinpath(str(traj_idx))
            log_path = os.path.join(save_dir, "combined_planar_pushing_logs.pkl")
            with open(log_path, "wb") as f:
                pickle.dump(combined, f)

    def _visualize_desired_slider_pose(self, t):
        # Visualizing the desired slider pose
        context = self._desired_position_source.GetMyContextFromRoot(self.context)
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
