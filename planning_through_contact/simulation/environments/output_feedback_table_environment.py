import logging
import os
from typing import Optional, List
import pathlib
import pickle

import numpy as np
from pydrake.all import (
    ConstantVectorSource,
    DiagramBuilder,
    LogVectorOutput,
    Meshcat,
    Simulator,
    Box as DrakeBox,
    RigidBody as DrakeRigidBody,
)

from planning_through_contact.planning.planar.planar_plan_config import (
    BoxWorkspace,
    PlanarPlanConfig,
    PlanarPushingWorkspace,
)

from planning_through_contact.geometry.planar.non_collision import (
    check_finger_pose_in_contact_location,
)

from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    CollisionGeometry,
    ContactLocation,
    PolytopeContactLocation,
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
    PlanarPushingLog,
    CombinedPlanarPushingLogs
)
from planning_through_contact.visualize.colors import COLORS

from planning_through_contact.experiments.utils import (
    get_default_plan_config,
)

logger = logging.getLogger(__name__)

# TODO: refactor
def _check_collision(
    pusher_pose_world: PlanarPose,
    slider_pose_world: PlanarPose,
    config: PlanarPlanConfig,
) -> bool:
    p_WP = pusher_pose_world.pos()
    R_WB = slider_pose_world.two_d_rot_matrix()
    p_WB = slider_pose_world.pos()

    # We need to compute the pusher pos in the frame of the slider
    p_BP = R_WB.T @ (p_WP - p_WB)
    pusher_pose_body = PlanarPose(p_BP[0, 0], p_BP[1, 0], 0)

    # we always add all non-collision modes, even when we don't add all contact modes
    # (think of maneuvering around the object etc)
    locations = [
        PolytopeContactLocation(ContactLocation.FACE, idx)
        for idx in range(config.slider_geometry.num_collision_free_regions)
    ]
    matching_locs = [
        loc
        for loc in locations
        if check_finger_pose_in_contact_location(pusher_pose_body, loc, config)
    ]
    if len(matching_locs) == 0:
        return True
    else:
        return False
    
def _slider_within_workspace(
    workspace: PlanarPushingWorkspace, pose: PlanarPose, slider: CollisionGeometry
) -> bool:
    """
    Checks whether the entire slider is within the workspace
    """
    R_WB = pose.two_d_rot_matrix()
    p_WB = pose.pos()

    p_Wv_s = [
        slider.get_p_Wv_i(vertex_idx, R_WB, p_WB).flatten()
        for vertex_idx in range(len(slider.vertices))
    ]

    lb, ub = workspace.slider.bounds
    vertices_within_workspace: bool = np.all([v <= ub for v in p_Wv_s]) and np.all(
        [v >= lb for v in p_Wv_s]
    )
    return vertices_within_workspace

def _get_slider_pose_within_workspace(
    workspace: PlanarPushingWorkspace,
    slider: CollisionGeometry,
    pusher_pose: PlanarPose,
    config: PlanarPlanConfig,
    limit_rotations: bool = False,
    enforce_entire_slider_within_workspace: bool = True,
) -> PlanarPose:
    valid_pose = False

    slider_pose = None
    while not valid_pose:
        x_initial = np.random.uniform(workspace.slider.x_min, workspace.slider.x_max)
        y_initial = np.random.uniform(workspace.slider.y_min, workspace.slider.y_max)
        EPS = 0.01
        if limit_rotations:
            # th_initial = np.random.uniform(-np.pi / 2 + EPS, np.pi / 2 - EPS)
            th_initial = np.random.uniform(-np.pi / 4 + EPS, np.pi / 4 - EPS)
        else:
            th_initial = np.random.uniform(-np.pi + EPS, np.pi - EPS)

        slider_pose = PlanarPose(x_initial, y_initial, th_initial)

        collides_with_pusher = _check_collision(pusher_pose, slider_pose, config)
        within_workspace = _slider_within_workspace(workspace, slider_pose, slider)

        if enforce_entire_slider_within_workspace:
            valid_pose = within_workspace and not collides_with_pusher
        else:
            valid_pose = not collides_with_pusher

    assert slider_pose is not None  # fix LSP errors

    return slider_pose

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
        
        self._plant = self._robot_system.get_station_plant()
        self._scene_graph = self._robot_system.get_scene_graph()
        self._slider = self._robot_system.get_slider()

        if self._multi_run_config:
            self._multi_run_idx = 0
            self._last_reset_time = 0.0
            self._total_runs = len(self._multi_run_config.initial_slider_poses)

            # used for reseting environment
            self._workspace = PlanarPushingWorkspace(
                slider=BoxWorkspace(
                        width=0.3, # 0.35,
                        height=0.4, # 0.5,
                        center=np.array([sim_config.slider_goal_pose.x, 
                                         sim_config.slider_goal_pose.y]),
                        buffer=0,
                    ),
                )
            self._plan_config = get_default_plan_config(
                slider_type='box' if sim_config.slider.name == 'box' else 'tee',
                pusher_radius=0.015,
                hardware=False,
            )

            
        
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

        # TODO (Adam): hacky way to get z value. Works for box and tee
        if self._sim_config.slider.name == "box":
            z_value = self._sim_config.slider.geometry.height / 2.0
        else: # T
            z_value = self._sim_config.slider.geometry.box_1.height / 2.0

        self._robot_state_to_rigid_transform = builder.AddNamedSystem(
            "RobotStateToRigidTransform",
            RobotStateToRigidTransform(
                self._plant,
                self._robot_system.robot_model_name,
            ),
        )

        self._meshcat = self._robot_system.get_meshcat()


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
        # if self._sim_config.collect_data:
        #     # Actual pusher state loggers
        #     pusher_pose_to_vector = builder.AddSystem(
        #         RigidTransformToPlanarPoseVectorSystem()
        #     )
        #     builder.Connect(
        #         self._robot_state_to_rigid_transform.GetOutputPort("pose"),
        #         pusher_pose_to_vector.get_input_port(),
        #     )
        #     self._pusher_pose_logger = LogVectorOutput(
        #         pusher_pose_to_vector.get_output_port(), builder
        #     )

        #     # Actual slider state loggers
        #     # TODO: after changing StateToRigidTransform to RobotStateToRigidTransform
        #     # this no longer works (since the slider input is generalized coords)
        #     slider_state_to_rigid_transform = builder.AddNamedSystem(
        #         "SliderStateToRigidTransform",
        #         StateToRigidTransform(
        #             self._plant, 
        #             self._robot_system.slider_model_name,
        #             z_value=z_value
        #         ),
        #     )
        #     slider_pose_to_vector = builder.AddSystem(
        #         RigidTransformToPlanarPoseVectorSystem()
        #     )
        #     builder.Connect(
        #         self._robot_system.GetOutputPort("object_state_measured"),
        #         slider_state_to_rigid_transform.GetInputPort("state"),
        #     )
        #     builder.Connect(
        #         slider_state_to_rigid_transform.GetOutputPort("pose"),
        #         slider_pose_to_vector.get_input_port(),
        #     )
        #     self._slider_pose_logger = LogVectorOutput(
        #         slider_pose_to_vector.get_output_port(), builder
        #     )

        #     # Desired pusher state loggers
        #     self._pusher_pose_desired_logger = LogVectorOutput(
        #         self._desired_position_source.GetOutputPort("planar_position_command"),
        #         builder,
        #     )
            
        #     # Desired slider state loggers
        #     desired_slider_source = builder.AddNamedSystem(
        #         "DesiredSliderSource",
        #         ConstantVectorSource(np.array([0.5, 0.0, 0.0]))
        #     )
        #     self._slider_pose_desired_logger = LogVectorOutput(
        #         desired_slider_source.get_output_port(),
        #         builder,
        #     )

        #     # Actual command loggers and desired command loggers are the same
        #     self._control_logger = LogVectorOutput(
        #         self._desired_position_source.GetOutputPort("planar_pose_command"),
        #         builder,
        #     )

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
    
    def set_pusher_planar_pose(self, pose: PlanarPose):
        q_v = np.array([pose.x, pose.y, 0.0, 0.0])
        self._plant.SetPositionsAndVelocities(self.mbp_context, 
                                 self._robot_model_instance, 
                                 q_v
        )

    def simulate(
        self,
        timeout=1e8,
        recording_file: Optional[str] = None,
        for_reset: bool = False,
    ):
        """
        :return: Returns a tuple of (success, simulation_time_s).
        """
        if recording_file:
            self._meshcat.StartRecording()
        time_step = self._sim_config.time_step * 10
        successful_idx = []
        if not isinstance(self._desired_position_source, TeleopPositionSource):
            for t in np.append(np.arange(0, timeout, time_step), timeout):
                self._simulator.AdvanceTo(t)
                # reset position if necessary
                reset_dict = self._should_reset_environment(t,
                                                            trans_tol=0.015,
                                                            rot_tol = 1.5*np.pi/180
                ) 
                if reset_dict['pusher'] or reset_dict['slider']:
                    if reset_dict['pusher'] == False and reset_dict['slider'] == True:
                        successful_idx.append(self._multi_run_idx-1)
                    if self._multi_run_idx == self._total_runs:
                        break
                    self._reset_environment(t, reset_dict)
                
                # visualization of target pose
                self._visualize_desired_slider_pose(
                    t,
                    self._sim_config.slider_goal_pose,
                    scale_factor=1.0
                )

                # Print every 5 seconds
                if t % 5 == 0:
                    # self._print_distance_to_target_pose()
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
        return successful_idx, save_dir
    
    def _print_distance_to_target_pose(self, 
                                       target_slider_pose: PlanarPose=PlanarPose(0.5, 0.0, 0.0)
    ):
        # Extract slider poses
        slider_position = self._plant.GetPositions(self.mbp_context, self._slider_model_instance)
        slider_pose = PlanarPose.from_generalized_coords(slider_position)
        
        # print distance to target pose
        x_error = target_slider_pose.x - slider_pose.x
        y_error = target_slider_pose.y - slider_pose.y
        theta_error = target_slider_pose.theta - slider_pose.theta
        print(f'\nx error: {100*x_error:.2f}cm')
        print(f'y error: {100*y_error:.2f}cm')
        print(f'orientation error: {theta_error*180.0/np.pi:.2f} degrees ({theta_error:.2f}rads)')


    def _should_reset_environment(self, 
                                  time: float,
                                  target_pusher_pose: PlanarPose=PlanarPose(0.5, 0.25, 0.0),
                                  target_slider_pose: PlanarPose=PlanarPose(0.5, 0.0, 0.0),
                                  trans_tol: float=0.02, # +/- 2cm
                                  rot_tol: float=2.0*np.pi/180, # +/- 2 degrees
        ) -> dict[str, bool]:
        false_dict = {'pusher': False, 'slider': False}
        if self._multi_run_config is None:
            return false_dict
        
        # Extract pusher and slider poses
        # TODO: need to do FK to get the slider pose
        pusher_position = self._plant.EvalBodyPoseInWorld(
            self.mbp_context,
            self._plant.GetBodyByName("pusher")
        ).translation()
        pusher_pose = PlanarPose(pusher_position[0], pusher_position[1], 0.0)
        slider_position = self._plant.GetPositions(self.mbp_context, self._slider_model_instance)
        slider_pose = PlanarPose.from_generalized_coords(slider_position)
        
        # Check if final pose has been reached
        reached_pusher_target_pose = target_pusher_pose.x-2*trans_tol <= pusher_pose.x <= target_pusher_pose.x+2*trans_tol and \
            target_pusher_pose.y-2*trans_tol <= pusher_pose.y <= target_pusher_pose.y+2*trans_tol

        if self._sim_config.slider.name == "box":
            reached_slider_target_pose = target_slider_pose.x-trans_tol <= slider_pose.x <= target_slider_pose.x+trans_tol and \
                target_slider_pose.y-trans_tol <= slider_pose.y <= target_slider_pose.y+trans_tol
        else:
            reached_slider_target_pose = target_slider_pose.x-trans_tol <= slider_pose.x <= target_slider_pose.x+trans_tol and \
                target_slider_pose.y-trans_tol <= slider_pose.y <= target_slider_pose.y+trans_tol and \
                target_slider_pose.theta-rot_tol <= slider_pose.theta <= target_slider_pose.theta+rot_tol

        if reached_pusher_target_pose and reached_slider_target_pose:
        # if reached_slider_target_pose:
            print(f"\n[Run {self._multi_run_idx}] Success! Reseting slider pose.")
            print("Initial pusher pose: ",
                    self._multi_run_config.initial_slider_poses[self._multi_run_idx-1])
            print("Final slider pose: ", slider_pose)
            return {'pusher': False, 'slider': True}
        
        if (time - self._last_reset_time) > self._multi_run_config.max_attempt_duration:
            print(f"\n[Run {self._multi_run_idx}] Reseting slider pose due to timeout.")
            print("Final pusher pose:", pusher_pose)
            print("Final slider pose:", slider_pose)
            return {'pusher': True, 'slider': True}
        else:
            return false_dict



    def _reset_environment(self, time, reset_dict) -> None:
        # reset pusher
        # if reset_dict['pusher']:
        #     self.set_pusher_planar_pose(PlanarPose(0.5, 0.25, 0.0))
        #     reset_position = np.array([0.5, 0.25])
        # else:
        #     reset_position = self._plant.GetPositions(self.mbp_context, self._robot_model_instance)
        
        # reset slider
        if reset_dict['slider']:
            # get a valid slider pose.
            slider_pose = self._multi_run_config.initial_slider_poses[self._multi_run_idx]
            slider_geometry = self._sim_config.dynamics_config.slider.geometry
            pusher_position = self._plant.EvalBodyPoseInWorld(
                self.mbp_context,
                self._plant.GetBodyByName("pusher")
            ).translation()
            pusher_pose = PlanarPose(pusher_position[0], pusher_position[1], 0.0)

            collides_with_pusher = _check_collision(pusher_pose, slider_pose, self._plan_config)
            within_workspace = _slider_within_workspace(self._workspace, slider_pose, slider_geometry)
            valid_pose = within_workspace and not collides_with_pusher

            if not valid_pose:
                slider_pose = _get_slider_pose_within_workspace(
                    self._workspace, 
                    slider_geometry, 
                    pusher_pose, 
                    self._plan_config
                )
            
            self.set_slider_planar_pose(slider_pose)
            self._multi_run_idx += 1
        
        self._last_reset_time = time


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

    def _visualize_desired_slider_pose(self, t, 
                                       desired_slider_pose: PlanarPose,
                                       scale_factor: float = 1.0):
        # Visualizing the desired slider pose
        self._robot_system._visualize_desired_slider_pose(
            desired_slider_pose,
            time_in_recording=t,
            scale_factor=scale_factor
        )