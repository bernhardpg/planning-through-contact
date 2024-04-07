import logging
import os
from typing import Optional
import numpy as np

import pickle
from pydrake.all import (
    DiagramBuilder,
    LogVectorOutput,
    Meshcat,
    Simulator,
    StateInterpolatorWithDiscreteDerivative,
    DiscreteTimeDelay,
)
from pydrake.systems.sensors import (
    ImageWriter,
    PixelType
)

from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.simulation.controllers.desired_planar_position_source_base import (
    DesiredPlanarPositionSourceBase,
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
from planning_through_contact.simulation.systems.planar_pose_to_generalized_coords import (
    PlanarPoseToGeneralizedCoords,
)
from planning_through_contact.visualize.analysis import (
    PlanarPushingLog,
)
from planning_through_contact.simulation.systems.planar_translation_to_rigid_transform_system import (
    PlanarTranslationToRigidTransformSystem,
)
from planning_through_contact.simulation.systems.diff_ik_system import DiffIKSystem
from planning_through_contact.simulation.sim_utils import LoadRobotOnly
from planning_through_contact.simulation.controllers.replay_position_source import (
    ReplayPositionSource,
)
from planning_through_contact.simulation.controllers.cylinder_actuated_station import (
    CylinderActuatedStation,
)

logger = logging.getLogger(__name__)


class DataCollectionTableEnvironment:
    def __init__(
        self,
        desired_position_source: DesiredPlanarPositionSourceBase,
        robot_system: RobotSystemBase,
        sim_config: PlanarPushingSimConfig,
        state_estimator_meshcat: Optional[Meshcat] = None,
    ):
        self._desired_position_source = desired_position_source
        self._robot_system = robot_system
        self._robot_model_name = robot_system.robot_model_name
        self._sim_config = sim_config
        self._meshcat = state_estimator_meshcat
        self._simulator = None
        self._save_dir = self._setup_data_dir(sim_config.data_dir)
        self._image_dir = f'{self._save_dir}/images'
        self._diff_ik_time_step = self._get_diff_ik_time_step()

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

        self._desired_state_source = builder.AddNamedSystem(
            "DesiredStateInterpolator",
            StateInterpolatorWithDiscreteDerivative(
                self._robot_system._num_positions,
                self._sim_config.time_step,
                suppress_initial_transient=True,
            ),
        )

        # IK systems
        z_value = sim_config.pusher_z_offset
        if type(self._robot_system) != CylinderActuatedStation:
            self._position_to_rigid_transform = builder.AddNamedSystem(
                "PlanarTranslationToRigidTransformSystem",
                PlanarTranslationToRigidTransformSystem(z_dist=z_value),
            )

            diff_ik_plant = LoadRobotOnly(sim_config, sim_config.scene_directive_name)
            self._diff_ik_system = builder.AddNamedSystem(
                "DiffIKSystem",
                DiffIKSystem(
                    plant=diff_ik_plant,
                    time_step=self._diff_ik_time_step,
                    default_joint_positions=sim_config.default_joint_positions,
                    disregard_angle=False,
                ),
            )

            state_size = diff_ik_plant.num_positions() + diff_ik_plant.num_velocities()
            self._time_delay = builder.AddSystem(
                DiscreteTimeDelay(
                    update_sec=self._diff_ik_time_step,
                    delay_time_steps=1,
                    vector_size=state_size
                ),
            )


        # Add system to convert slider_pose to generalized coords
        self._slider_pose_to_generalized_coords = builder.AddNamedSystem(
            "PlanarPoseToGeneralizedCoords",
            PlanarPoseToGeneralizedCoords(
                z_value=z_value,
                z_axis_is_positive=True,
            ),
        )

        
        if type(self._robot_system) == CylinderActuatedStation:
            # No diff IK required for actuated cylinder
            builder.Connect(
                self._desired_position_source.GetOutputPort("planar_position_command"),
                self._desired_state_source.get_input_port(),
            )

            builder.Connect(
                self._desired_state_source.get_output_port(),
                self._state_estimator.GetInputPort("robot_state"),
            )
        else:
            # Diff IK connections
            builder.Connect(
                self._desired_position_source.GetOutputPort("planar_position_command"),
                self._position_to_rigid_transform.GetInputPort("vector_input"),
            )

            builder.Connect(
                self._position_to_rigid_transform.GetOutputPort("rigid_transform_output"),
                self._diff_ik_system.GetInputPort("rigid_transform_input")
            )

            builder.Connect(
                self._diff_ik_system.get_output_port(),
                self._desired_state_source.get_input_port(),
            )

            builder.Connect(
                self._desired_state_source.get_output_port(),
                self._state_estimator.GetInputPort("robot_state"),
            )

            builder.Connect(
                self._state_estimator.GetOutputPort(f"{self._robot_model_name}_state"),
                self._time_delay.get_input_port()
            )

            builder.Connect(
                self._time_delay.get_output_port(),
                self._diff_ik_system.GetInputPort("state")
            )

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

        if sim_config.collect_data:
            # Set up camera logging
            # TODO: add image writer per camera
            assert sim_config.camera_configs is not None
                                    
            image_writer_system = ImageWriter()
            image_writer_system.DeclareImageInputPort(
                pixel_type=PixelType.kRgba8U,
                port_name="overhead_camera_image",
                file_name_format= self._image_dir + '/{time_msec}.png',
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

            # Set up desired pusher planar pose loggers
            self._pusher_pose_desired_logger = LogVectorOutput(
                self._desired_position_source.GetOutputPort("planar_position_command"),
                builder,
            )
            

        diagram = builder.Build()
        self._diagram = diagram

        self._simulator = Simulator(diagram)
        if sim_config.use_realtime:
            self._simulator.set_target_realtime_rate(1.0)
        
        self.context = self._simulator.get_mutable_context()

    def _get_diff_ik_time_step(self):
        if type(self._desired_position_source) == ReplayPositionSource and \
            self._desired_position_source.get_time_step() is not None:
            return self._desired_position_source.get_time_step()
        return self._sim_config.time_step

    def simulate(
        self,
        timeout=1e8,
        recording_file: Optional[str] = None,
    ) -> None:
        """
        :return: Returns a tuple of (success, simulation_time_s).
        """
        assert not isinstance(self._desired_position_source, TeleopPositionSource)

        if recording_file:
            self._state_estimator.meshcat.StartRecording()
            self._meshcat.StartRecording()

        time_step = self._sim_config.time_step * 10
        for t in np.append(np.arange(0, timeout, time_step), timeout):
            self._simulator.AdvanceTo(t)
            self._visualize_desired_slider_pose(t)
            # Print the time every 5 seconds
            if t % 5 == 0:
                logger.info(f"t={t}")

        else:
            self._simulator.AdvanceTo(timeout)
        self.save_logs(recording_file, self._save_dir)
        self.save_data()

    def export_diagram(self, filename: str):
        import pydot

        pydot.graph_from_dot_data(self._diagram.GetGraphvizString())[0].write_pdf(  # type: ignore
            filename
        )
        print(f"Saved diagram to: {filename}")

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

    def save_data(self):
        """
        This function only logs the desired pusher information (since this
        is all that is needed to train diffusion policies).
        To log additional information, refer to table_environment.py
        """
        if self._sim_config.collect_data:
            assert self._sim_config.data_dir is not None

            # Save the logs
            pusher_pose_desired_log = self._pusher_pose_desired_logger.FindLog(
                self.context
            )
            pusher_desired = PlanarPushingLog.from_pose_vector_log(
                pusher_pose_desired_log
            )

            log_path = os.path.join(self._save_dir, "planar_position_command.pkl")
            with open(log_path, "wb") as f:
                pickle.dump(pusher_desired, f)

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
    
    def _setup_data_dir(self, data_dir: str) -> str:
        assert data_dir is not None
        # Create data_dir if it doesn't already exist
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)

        # Find the next available trajectory index
        traj_idx = 0
        for path in os.listdir(data_dir):
            if os.path.isdir(os.path.join(data_dir, path)):
                traj_idx += 1
        
        # Setup the current directory
        os.makedirs(f'{data_dir}/{traj_idx}/images')
        return f'{data_dir}/{traj_idx}'