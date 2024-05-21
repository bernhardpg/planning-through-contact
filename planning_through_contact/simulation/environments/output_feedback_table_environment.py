import logging
import os
import pathlib
from enum import Enum
from typing import Optional

import numpy as np
from pydrake.all import DiagramBuilder, LogVectorOutput, Meshcat, Rgba, Simulator

from planning_through_contact.experiments.utils import get_default_plan_config
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.planning.planar.planar_plan_config import (
    BoxWorkspace,
    PlanarPushingWorkspace,
)
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
from planning_through_contact.simulation.sim_utils import (
    check_collision,
    create_goal_geometries,
    get_slider_pose_within_workspace,
    slider_within_workspace,
    visualize_desired_slider_pose,
)
from planning_through_contact.simulation.systems.rigid_transform_to_planar_pose_vector_system import (
    RigidTransformToPlanarPoseVectorSystem,
)
from planning_through_contact.simulation.systems.robot_state_to_rigid_transform import (
    RobotStateToRigidTransform,
)

logger = logging.getLogger(__name__)


class ResetStatus(Enum):
    RESET_SUCCESS = "reset_success"
    RESET_TIMEOUT = "reset_timeout"
    NO_RESET = "no_reset"


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
        self._goal_geometries = []

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
                    width=0.3,  # 0.35,
                    height=0.4,  # 0.5,
                    center=np.array(
                        [sim_config.slider_goal_pose.x, sim_config.slider_goal_pose.y]
                    ),
                    buffer=0,
                ),
            )
            self._plan_config = get_default_plan_config(
                slider_type="box" if sim_config.slider.name == "box" else "tee",
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
        self._plant.SetPositionsAndVelocities(
            self.mbp_context, self._robot_model_instance, q_v
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
                reset_status = self._should_reset_slider(
                    t,
                    target_slider_pose=self._sim_config.slider_goal_pose,
                    target_pusher_pose=self._sim_config.pusher_start_pose,
                    evaluate_final_pusher_position=self._multi_run_config.evaluate_final_pusher_position,
                    evaluate_final_slider_rotation=self._multi_run_config.evaluate_final_slider_rotation,
                    trans_tol=self._multi_run_config.trans_tol,
                    rot_tol=self._multi_run_config.rot_tol * np.pi / 180,
                )
                if reset_status != ResetStatus.NO_RESET:
                    if reset_status == ResetStatus.RESET_SUCCESS:
                        successful_idx.append(self._multi_run_idx - 1)
                    if self._multi_run_idx == self._total_runs:
                        break
                    self._reset_slider(t)

                # visualization of target pose
                if len(self._goal_geometries) == 0:
                    self._goal_geometries = create_goal_geometries(
                        self._robot_system,
                        self._sim_config.slider_goal_pose,
                    )
                visualize_desired_slider_pose(
                    self._robot_system,
                    self._sim_config.slider_goal_pose,
                    self._goal_geometries,
                    time_in_recording=t,
                )

                # Print every 5 seconds
                if t % 5 == 0:
                    # self._print_distance_to_target_pose()
                    logger.info(f"t={t}")

        else:
            self._simulator.AdvanceTo(timeout)

        traj_idx = 0
        if os.path.exists(self._sim_config.log_dir):
            for path in os.listdir(self._sim_config.log_dir):
                if os.path.isdir(os.path.join(self._sim_config.log_dir, path)):
                    traj_idx += 1
        os.makedirs(os.path.join(self._sim_config.log_dir, str(traj_idx)))
        save_dir = pathlib.Path(self._sim_config.log_dir).joinpath(str(traj_idx))

        self.save_logs(recording_file, save_dir)
        # self.save_data(save_dir)
        return successful_idx, save_dir

    def _print_distance_to_target_pose(
        self, target_slider_pose: PlanarPose = PlanarPose(0.5, 0.0, 0.0)
    ):
        # Extract slider poses
        slider_position = self._plant.GetPositions(
            self.mbp_context, self._slider_model_instance
        )
        slider_pose = PlanarPose.from_generalized_coords(slider_position)

        # print distance to target pose
        x_error = target_slider_pose.x - slider_pose.x
        y_error = target_slider_pose.y - slider_pose.y
        theta_error = target_slider_pose.theta - slider_pose.theta
        print(f"\nx error: {100*x_error:.2f}cm")
        print(f"y error: {100*y_error:.2f}cm")
        print(
            f"orientation error: {theta_error*180.0/np.pi:.2f} degrees ({theta_error:.2f}rads)"
        )

    def _should_reset_slider(
        self,
        time: float,
        target_pusher_pose: PlanarPose,
        target_slider_pose: PlanarPose,
        evaluate_final_pusher_position: bool = True,
        evaluate_final_slider_rotation: bool = True,
        trans_tol: float = 0.01,  # +/- 2cm
        rot_tol: float = 2.0 * np.pi / 180,  # +/- 2 degrees
    ):
        if self._multi_run_config is None:
            return False

        # Extract pusher and slider poses
        pusher_position = self._plant.EvalBodyPoseInWorld(
            self.mbp_context, self._plant.GetBodyByName("pusher")
        ).translation()
        pusher_pose = PlanarPose(pusher_position[0], pusher_position[1], 0.0)
        slider_position = self._plant.GetPositions(
            self.mbp_context, self._slider_model_instance
        )
        slider_pose = PlanarPose.from_generalized_coords(slider_position)

        # Evaluate pusher pose
        if evaluate_final_pusher_position:
            reached_pusher_target_pose = (
                target_pusher_pose.x - 2 * trans_tol
                <= pusher_pose.x
                <= target_pusher_pose.x + 2 * trans_tol
                and target_pusher_pose.y - 2 * trans_tol
                <= pusher_pose.y
                <= target_pusher_pose.y + 2 * trans_tol
            )
        else:
            reached_pusher_target_pose = True

        # Evaluate slider pose
        reached_slider_target_pose = (
            target_slider_pose.x - trans_tol
            <= slider_pose.x
            <= target_slider_pose.x + trans_tol
            and target_slider_pose.y - trans_tol
            <= slider_pose.y
            <= target_slider_pose.y + trans_tol
        )
        if evaluate_final_slider_rotation:
            reached_slider_target_pose = (
                reached_slider_target_pose
                and target_slider_pose.theta - rot_tol
                <= slider_pose.theta
                <= target_slider_pose.theta + rot_tol
            )

        if reached_pusher_target_pose and reached_slider_target_pose:
            print(f"\n[Run {self._multi_run_idx}] Success! Reseting slider pose.")
            print(
                "Initial pusher pose: ",
                self._multi_run_config.initial_slider_poses[self._multi_run_idx - 1],
            )
            print("Final slider pose: ", slider_pose)
            return ResetStatus.RESET_SUCCESS
        elif (
            time - self._last_reset_time
        ) > self._multi_run_config.max_attempt_duration:
            print(f"\n[Run {self._multi_run_idx}] Reseting slider pose due to timeout.")
            print("Final pusher pose:", pusher_pose)
            print("Final slider pose:", slider_pose)
            return ResetStatus.RESET_TIMEOUT
        else:
            return ResetStatus.NO_RESET

    def _reset_slider(self, time) -> None:
        # Extract variables for collision checking
        slider_geometry = self._sim_config.dynamics_config.slider.geometry
        pusher_position = self._plant.EvalBodyPoseInWorld(
            self.mbp_context, self._plant.GetBodyByName("pusher")
        ).translation()
        pusher_pose = PlanarPose(pusher_position[0], pusher_position[1], 0.0)

        # Determine slider reset pose
        slider_pose = self._multi_run_config.initial_slider_poses[self._multi_run_idx]
        collides_with_pusher = check_collision(
            pusher_pose, slider_pose, self._plan_config
        )
        within_workspace = slider_within_workspace(
            self._workspace, slider_pose, slider_geometry
        )
        valid_pose = within_workspace and not collides_with_pusher

        if not valid_pose:
            slider_pose = get_slider_pose_within_workspace(
                self._workspace, slider_geometry, pusher_pose, self._plan_config
            )

        self.set_slider_planar_pose(slider_pose)
        self._multi_run_idx += 1
        self._last_reset_time = time

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
