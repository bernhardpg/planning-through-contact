import logging
import os
from typing import Optional, List
import pathlib
import pickle

import numpy as np
from pydrake.all import (
    ConstantVectorSource,
    Demultiplexer,
    DiagramBuilder,
    LogVectorOutput,
    Meshcat,
    Simulator,
    Box as DrakeBox,
    RigidBody as DrakeRigidBody,
    GeometryInstance,
    MakePhongIllustrationProperties,
    Rgba,
    MultibodyPlant,
    SceneGraph,
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
from planning_through_contact.simulation.systems.state_to_rigid_transform import (
    StateToRigidTransform,
)
from planning_through_contact.visualize.analysis import (
    plot_joint_state_logs,
    plot_and_save_planar_pushing_logs_from_sim,
    PlanarPushingLog,
    CombinedPlanarPushingLogs
)
from planning_through_contact.visualize.colors import COLORS

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
        self._multi_run_config = sim_config.multi_run_config
        self._meshcat = station_meshcat
        self._simulator = None
        
        self._plant = self._robot_system.station_plant
        self._scene_graph = self._robot_system._scene_graph
        self._slider = self._robot_system.slider
        
        if self._multi_run_config:
            self._multi_run_idx = 0
            self._last_reset_time = 0.0
            self._total_runs = len(self._multi_run_config.initial_slider_poses)
        
        self._robot_model_instance = self._plant.GetModelInstanceByName(
            self._robot_system.robot_model_name
        )
        self._slider_model_instance = self._plant.GetModelInstanceByName(
            self._robot_system.slider_model_name
        )

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
            "PusherStateToRigidTransform",
            StateToRigidTransform(
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
            self._robot_state_to_rigid_transform.GetInputPort("state"),
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

        # Add loggers
        if self._sim_config.collect_data:
            # Actual pusher state loggers
            pusher_pose_to_vector = builder.AddSystem(
                RigidTransformToPlanarPoseVectorSystem()
            )
            builder.Connect(
                self._robot_state_to_rigid_transform.GetOutputPort("pose"),
                pusher_pose_to_vector.get_input_port(),
            )
            self._pusher_pose_logger = LogVectorOutput(
                pusher_pose_to_vector.get_output_port(), builder
            )

            # Actual slider state loggers
            slider_state_to_rigid_transform = builder.AddNamedSystem(
                "SliderStateToRigidTransform",
                StateToRigidTransform(
                    self._robot_system.station_plant, 
                    self._robot_system.slider_model_name,
                    z_value=z_value
                ),
            )
            slider_pose_to_vector = builder.AddSystem(
                RigidTransformToPlanarPoseVectorSystem()
            )
            builder.Connect(
                self._robot_system.GetOutputPort("object_state_measured"),
                slider_state_to_rigid_transform.GetInputPort("state"),
            )
            builder.Connect(
                slider_state_to_rigid_transform.GetOutputPort("pose"),
                slider_pose_to_vector.get_input_port(),
            )
            self._slider_pose_logger = LogVectorOutput(
                slider_pose_to_vector.get_output_port(), builder
            )

            # Desired pusher state loggers
            self._pusher_pose_desired_logger = LogVectorOutput(
                self._desired_position_source.GetOutputPort("planar_position_command"),
                builder,
            )
            
            # Desired slider state loggers
            desired_slider_source = builder.AddNamedSystem(
                "DesiredSliderSource",
                ConstantVectorSource(np.array([0.5, 0.0, 0.0]))
            )
            self._slider_pose_desired_logger = LogVectorOutput(
                desired_slider_source.get_output_port(),
                builder,
            )

            # Actual command loggers and desired command loggers are the same
            self._control_logger = LogVectorOutput(
                self._desired_position_source.GetOutputPort("planar_pose_command"),
                builder,
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
        if self._multi_run_config:
            self.set_slider_planar_pose(
                self._multi_run_config.initial_slider_poses[self._multi_run_idx]
            )
            self._multi_run_idx += 1
        else:
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
                # reset position if necessary
                if self._should_reset_environment(t):
                    self._reset_environment(t)
                self._simulator.AdvanceTo(t)
                # self._visualize_desired_slider_pose(
                #     self._sim_config.slider_goal_pose
                # )
                # Print the time every 5 seconds
                if t % 5 == 0:
                    logger.info(f"t={t}")

        else:
            self._simulator.AdvanceTo(timeout)

        traj_idx = 0
        if os.path.exists(self._sim_config.data_dir):
            for path in os.listdir(self._sim_config.data_dir):
                if os.path.isdir(os.path.join(self._sim_config.data_dir, path)):
                    traj_idx += 1
        os.makedirs(os.path.join(self._sim_config.data_dir, str(traj_idx)))
        save_dir = pathlib.Path(self._sim_config.data_dir).joinpath(str(traj_idx))
        
        self.save_logs(recording_file, save_dir)
        self.save_data(save_dir)
    
    def _should_reset_environment(self, 
                                  time: float,
                                  target_pusher_pose: PlanarPose=PlanarPose(0.5, 0.25, 0.0),
                                  target_slider_pose: PlanarPose=PlanarPose(0.5, 0.0, 0.0),
                                  trans_tol: float=0.02, # +/- 2cm
                                  rot_tol: float=2.0*np.pi/180, # +/- 2 degrees
                                  slider_vel_tol: float=0.008 # 8mm/s
        ) -> bool:
        if self._multi_run_config is None:
            return False
        
        # Extract pusher and slider poses
        pusher_position = self._plant.GetPositions(self.mbp_context, self._robot_model_instance)
        pusher_speed = np.linalg.norm(
            self._plant.GetVelocities(self.mbp_context, self._robot_model_instance)
        )
        pusher_pose = PlanarPose(*pusher_position, 0.0)
        slider_position = self._plant.GetPositions(self.mbp_context, self._slider_model_instance)
        slider_pose = PlanarPose.from_generalized_coords(slider_position)
        
        # Check if final pose has been reached
        reached_pusher_target_pose = target_pusher_pose.x-2*trans_tol <= pusher_pose.x <= target_pusher_pose.x+2*trans_tol and \
            target_pusher_pose.y-2*trans_tol <= pusher_pose.y <= target_pusher_pose.y+2*trans_tol and \
            np.linalg.norm(pusher_speed) <= slider_vel_tol

        reached_slider_target_pose = target_slider_pose.x-trans_tol <= slider_pose.x <= target_slider_pose.x+trans_tol and \
            target_slider_pose.y-trans_tol <= slider_pose.y <= target_slider_pose.y+trans_tol and \
            target_slider_pose.theta-rot_tol <= slider_pose.theta <= target_slider_pose.theta+rot_tol

        if reached_pusher_target_pose and reached_slider_target_pose:
            print("Success! Reseting slider pose.")
            print("Initial pusher pose: ", 
                  self._multi_run_config.initial_slider_poses[self._multi_run_idx])
            print("Final slider pose: ", slider_pose)
        
        if self._multi_run_idx >= self._total_runs:
            return False
        
        if reached_pusher_target_pose and reached_slider_target_pose:
            return True
        
        if (time - self._last_reset_time) > self._multi_run_config.max_attempt_duration:
            print("Reseting slider pose due to timeout.")
            print("Final pusher pose:", pusher_pose)
            print("Final pusher speed:", pusher_speed)
            print("Final slider pose:", slider_pose)
            return True
        else:
            return False



    def _reset_environment(self, time) -> None:
        self.set_slider_planar_pose(self._multi_run_config.initial_slider_poses[self._multi_run_idx])
        self._last_reset_time = time
        self._multi_run_idx += 1

    def save_data(self, save_dir):
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

            pusher_actual = PlanarPushingLog.from_pose_vector_log(pusher_pose_log)
            slider_actual = PlanarPushingLog.from_log(slider_pose_log, control_log)
            pusher_desired = PlanarPushingLog.from_pose_vector_log(
                pusher_pose_desired_log
            )
            slider_desired = PlanarPushingLog.from_log(
                slider_pose_desired_log,
                control_log,
            )
            # TODO: didn't actually need to save it in this format
            # actually want to save as PlanarPushingTrajectory
            # and call its save method
            # Worry about this later
            combined = CombinedPlanarPushingLogs(
                pusher_actual=pusher_actual,
                slider_actual=slider_actual,
                pusher_desired=pusher_desired,
                slider_desired=slider_desired,
            )

            # assumes that a directory for this trajectory has already been
            # created (when saving the images)
            log_path = os.path.join(save_dir, "combined_planar_pushing_logs.pkl")
            print(f"Saving combined logs to {log_path}")
            with open(log_path, "wb") as f:
                pickle.dump(combined, f)
    
    def save_logs(self, recording_file: Optional[str], save_dir: str):
        if recording_file:
            self._meshcat.StopRecording()
            self._meshcat.SetProperty("/drake/contact_forces", "visible", False)
            self._meshcat.PublishRecording()
            res = self._meshcat.StaticHtml()
            if save_dir:
                recording_file = os.path.join(save_dir, recording_file)
            with open(recording_file, "w") as f:
                f.write(res)

    # # TODO: fix this
    # def _visualize_desired_slider_pose(
    #     self, desired_planar_pose: PlanarPose,
    # ) -> None:
    #     shapes = self.get_slider_shapes()
    #     poses = self.get_slider_shape_poses()

    #     heights = [shape.height() for shape in shapes]
    #     min_height = min(heights)
    #     desired_pose = desired_planar_pose.to_pose(
    #         min_height / 2, z_axis_is_positive=True
    #     )

    #     source_id = self._scene_graph.RegisterSource()
    #     BOX_COLOR = COLORS["emeraldgreen"]
    #     DESIRED_POSE_ALPHA = 0.4
    #     for idx, (shape, pose) in enumerate(zip(shapes, poses)):
    #         geom_instance = GeometryInstance(
    #             # desired_pose.multiply(pose),
    #             desired_pose,
    #             shape,
    #             f"shape_{idx}",
    #         )
    #         curr_shape_geometry_id = self._scene_graph.RegisterAnchoredGeometry(
    #             source_id,
    #             geom_instance,
    #         )
    #         # self._scene_graph.AssignRole(
    #         #     source_id,
    #         #     curr_shape_geometry_id,
    #         #     MakePhongIllustrationProperties(
    #         #         BOX_COLOR.diffuse(DESIRED_POSE_ALPHA)
    #         #     ),
    #         # )
    #         geom_name = f"goal_shape_{idx}"
    #         self._meshcat.SetObject(
    #             geom_name, shape, rgba=Rgba(*BOX_COLOR.diffuse(DESIRED_POSE_ALPHA))
    #         )
    #         break

    # def get_slider_shapes(self) -> List[DrakeBox]:
    #     slider_body = self.get_slider_body()
    #     collision_geometries_ids = self._plant.GetCollisionGeometriesForBody(
    #         slider_body
    #     )

    #     inspector = self._scene_graph.model_inspector()
    #     shapes = [inspector.GetShape(id) for id in collision_geometries_ids]

    #     # for now we only support Box shapes
    #     assert all([isinstance(shape, DrakeBox) for shape in shapes])

    #     return shapes
    
    # def get_slider_body(self) -> DrakeRigidBody:
    #     slider_body = self._plant.GetUniqueFreeBaseBodyOrThrow(self._slider)
    #     return slider_body

    # def get_slider_shape_poses(self) -> List[DrakeBox]:
    #     slider_body = self.get_slider_body()
    #     collision_geometries_ids = self._plant.GetCollisionGeometriesForBody(
    #         slider_body
    #     )

    #     inspector = self._scene_graph.model_inspector()
    #     poses = [inspector.GetPoseInFrame(id) for id in collision_geometries_ids]

    #     return poses