import logging
import os
from typing import Optional
import numpy as np
from math import ceil
from dataclasses import dataclass

import pickle
from pydrake.all import (
    DiagramBuilder,
    LogVectorOutput,
    Meshcat,
    Simulator,
    StateInterpolatorWithDiscreteDerivative,
    DiscreteTimeDelay,
    ZeroOrderHold
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

@dataclass
class PlanConfig:
    # Geometry
    slider_type: str
    pusher_radius: float

    # Solver
    contact_lam_min: float
    contact_lam_max: float
    distance_to_object_socp: float

    # Workspace
    width: float
    height: float
    center: np.ndarray
    buffer: float

    # Plan generation
    seed: int
    num_plans: int
    pusher_start_pose: PlanarPose
    slider_goal_pose: PlanarPose
    limit_rotations: bool
    noise_final_pose: float

class DataCollectionConfig:
    def __init__(
        self,
        generate_plans: bool,
        render_plans: bool,
        convert_to_zarr: bool,
        convert_to_zarr_reduce: bool,
        plans_dir: str,
        rendered_plans_dir: str,
        zarr_path: str,
        state_chunk_length: int,
        action_chunk_length: int,
        target_chunk_length: int,
        image_chunk_length: int,
        policy_freq: float,
        plan_config: PlanConfig,
        LLSUB_RANK: int = None,
        LLSUB_SIZE: int = None,    
    ):
        # Data collection steps
        self.generate_plans = generate_plans
        self.render_plans = render_plans
        self.convert_to_zarr = convert_to_zarr
        self.convert_to_zarr_reduce = convert_to_zarr_reduce

        # Data collection directories
        self.plans_dir = plans_dir
        self.rendered_plans_dir = rendered_plans_dir
        self.zarr_path = zarr_path

        # zarr params
        self.state_chunk_length = state_chunk_length
        self.action_chunk_length = action_chunk_length
        self.target_chunk_length = target_chunk_length
        self.image_chunk_length = image_chunk_length
        self.policy_freq = policy_freq

        # Plan generatino config
        self.plan_config = plan_config

        # Supercloud settings
        self.LLSUB_RANK = LLSUB_RANK
        self.LLSUB_SIZE = LLSUB_SIZE
        if self.LLSUB_RANK is not None and self.LLSUB_SIZE is not None:
            assert not self.convert_to_zarr and not self.convert_to_zarr_reduce
            
            self.plans_dir = f"{self.plans_dir}/run_{self.LLSUB_RANK}"
            self.rendered_plans_dir = f"{self.rendered_plans_dir}/run_{self.LLSUB_RANK}"
            self.plan_config.seed += self.LLSUB_RANK
            num_plans_per_run = ceil(1.0*self.plan_config.num_plans / self.LLSUB_SIZE)
            if self.LLSUB_RANK != self.LLSUB_SIZE - 1:
                self.plan_config.num_plans = num_plans_per_run
            else:
                num_plans = self.plan_config.num_plans - num_plans_per_run*(self.LLSUB_SIZE - 1)
                self.plan_config.num_plans = num_plans


class DataCollectionTableEnvironment:
    def __init__(
        self,
        desired_position_source: DesiredPlanarPositionSourceBase,
        robot_system: RobotSystemBase,
        sim_config: PlanarPushingSimConfig,
        data_collection_config: DataCollectionConfig,
        state_estimator_meshcat: Optional[Meshcat] = None,
    ):
        self._desired_position_source = desired_position_source
        self._robot_system = robot_system
        self._robot_model_name = robot_system.robot_model_name
        self._sim_config = sim_config
        self._meshcat = state_estimator_meshcat
        self._simulator = None
        self._data_collection_config = data_collection_config
        self._data_collection_dir = self._setup_data_collection_dir(
            data_collection_config.rendered_plans_dir
        )
        self._image_dir = f'{self._data_collection_dir}/images'
        self._log_path = f'{self._data_collection_dir}/log.txt'
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
                self._diff_ik_time_step,
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
                    log_path=self._log_path,
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

            self._diff_ik_zoh = builder.AddSystem(
                ZeroOrderHold(
                    period_sec = self._diff_ik_time_step,
                    vector_size = diff_ik_plant.num_positions()
                )
            )

        # Add system to convert slider_pose to generalized coords
        self._slider_pose_to_generalized_coords = builder.AddNamedSystem(
            "PlanarPoseToGeneralizedCoords",
            PlanarPoseToGeneralizedCoords(
                z_value=0.025, # Assumes objects are 5cm tall
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
                self._position_to_rigid_transform.GetInputPort("planar_position_input"),
            )

            builder.Connect(
                self._position_to_rigid_transform.GetOutputPort("rigid_transform_output"),
                self._diff_ik_system.GetInputPort("rigid_transform_input")
            )

            builder.Connect(
                self._diff_ik_system.get_output_port(),
                self._diff_ik_zoh.get_input_port(),
            )

            builder.Connect(
                self._diff_ik_zoh.get_output_port(),
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

        # Set up camera logging
        image_writers = []
        for camera_config in sim_config.camera_configs:                
            image_writers.append(ImageWriter())
            image_writers[-1].DeclareImageInputPort(
                pixel_type=PixelType.kRgba8U,
                port_name=f"{camera_config.name}_image",
                file_name_format= self._image_dir + '/{time_msec}.png',
                publish_period=0.1,
                start_time=0.0
            )
            builder.AddNamedSystem(
                f"{camera_config}_image_writer",
                image_writers[-1]
            )
            builder.Connect(
                self._state_estimator.GetOutputPort(
                    f"rgbd_sensor_state_estimator_{camera_config.name}"
                ),
                image_writers[-1].get_input_port()
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
        else:
            self._simulator.AdvanceTo(timeout)
        self.save_logs(recording_file, self._data_collection_dir)
        self.save_data()

    def export_diagram(self, filename: str):
        import pydot

        pydot.graph_from_dot_data(self._diagram.GetGraphvizString())[0].write_pdf(  # type: ignore
            filename
        )
        print(f"Saved diagram to: {filename}")

    def save_logs(self, recording_file: Optional[str], save_dir: str):
        if recording_file:
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
        assert self._data_collection_dir is not None

        # Save the logs
        pusher_pose_desired_log = self._pusher_pose_desired_logger.FindLog(
            self.context
        )
        pusher_desired = PlanarPushingLog.from_pose_vector_log(
            pusher_pose_desired_log
        )

        log_path = os.path.join(self._data_collection_dir, "planar_position_command.pkl")
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
    
    def _setup_data_collection_dir(self, data_collection_dir: str) -> str:
        assert data_collection_dir is not None
        # Create data_collection_dir if it doesn't already exist
        if not os.path.exists(data_collection_dir):
            os.makedirs(data_collection_dir)

        # Find the next available trajectory index
        traj_idx = 0
        for path in os.listdir(data_collection_dir):
            if os.path.isdir(os.path.join(data_collection_dir, path)):
                traj_idx += 1
        
        # Setup the current directory
        os.makedirs(f'{data_collection_dir}/{traj_idx}/images')
        open(f'{data_collection_dir}/{traj_idx}/log.txt', 'w').close()
        return f'{data_collection_dir}/{traj_idx}'
    
    def _get_diff_ik_time_step(self):
        if type(self._desired_position_source) == ReplayPositionSource and \
            self._desired_position_source.get_time_step() is not None:
            return self._desired_position_source.get_time_step()
        return self._sim_config.time_step