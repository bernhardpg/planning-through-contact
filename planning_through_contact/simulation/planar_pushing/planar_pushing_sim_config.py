from dataclasses import dataclass, field
from typing import List, Optional

import hydra
import numpy as np
import numpy.typing as npt
from omegaconf import OmegaConf
from pydrake.all import Rgba, RollPitchYaw
from pydrake.common.schema import Transform
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.multibody.plant import ContactModel
from pydrake.systems.sensors import CameraConfig

from planning_through_contact.experiments.utils import (
    get_box,
    get_default_plan_config,
    get_tee,
)
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.planar.planar_pushing_trajectory import (
    PlanarPushingTrajectory,
)
from planning_through_contact.geometry.rigid_body import RigidBody
from planning_through_contact.planning.planar.planar_plan_config import (
    BoxWorkspace,
    PlanarPushingWorkspace,
    SliderPusherSystemConfig,
)
from planning_through_contact.simulation.controllers.diffusion_policy_source import (
    DiffusionPolicyConfig,
)
from planning_through_contact.simulation.controllers.hybrid_mpc import HybridMpcConfig
from planning_through_contact.simulation.sim_utils import get_slider_start_poses

from planning_through_contact.simulation.controllers.diffusion_policy_source import DiffusionPolicyConfig
from planning_through_contact.experiments.utils import (
    get_box,
    get_tee,
    get_default_plan_config,
    get_arbitrary,
)
from planning_through_contact.simulation.sim_utils import (
    get_slider_start_poses,
)
from planning_through_contact.tools.utils import PhysicalProperties

class MultiRunConfig:
    def __init__(
            self,
            num_runs: int, 
            max_attempt_duration: float, 
            seed: int, 
            slider_type: str,
            arbitrary_shape_pickle_path: str,
            pusher_start_pose: PlanarPose,
            slider_goal_pose: PlanarPose,
            workspace_width: float,
            workspace_height: float,
            trans_tol: float=0.01,
            rot_tol: float=0.01, # degrees
            evaluate_final_pusher_position: bool=True,
            evaluate_final_slider_rotation: bool=True,
            slider_physical_properties: PhysicalProperties=None
    ):
        # Set up multi run config
        config = get_default_plan_config(
            slider_type=slider_type,
            arbitrary_shape_pickle_path=arbitrary_shape_pickle_path,
            pusher_radius=0.015,
            hardware=False,
            slider_physical_properties=slider_physical_properties,
        )
        # update config (probably don't need these)
        config.contact_config.lam_min = 0.15
        config.contact_config.lam_max = 0.85
        config.non_collision_cost.distance_to_object_socp = 0.25

        # Get initial slider poses
        workspace = PlanarPushingWorkspace(
            slider=BoxWorkspace(
                width=workspace_width,
                height=workspace_height,
                center=np.array([slider_goal_pose.x, slider_goal_pose.y]),
                buffer=0,
            ),
        )
        self.initial_slider_poses = get_slider_start_poses(
            seed=seed,
            num_plans=num_runs,
            workspace=workspace,
            config=config,
            pusher_pose=pusher_start_pose,
            limit_rotations=False,
        )
        self.num_runs = num_runs
        self.seed = seed
        self.target_slider_poses = [slider_goal_pose] * num_runs
        self.max_attempt_duration = max_attempt_duration
        self.trans_tol = trans_tol
        self.rot_tol = rot_tol
        self.evaluate_final_pusher_position = evaluate_final_pusher_position
        self.evaluate_final_slider_rotation = evaluate_final_slider_rotation

    def __str__(self):
        slider_pose_str = f"initial_slider_poses: {self.initial_slider_poses}"
        target_pose_str = f"target_slider_poses: {self.target_slider_poses}"
        return f"{slider_pose_str}\n{target_pose_str}\nmax_attempt_duration: {self.max_attempt_duration}"

    def __eq__(self, other: "MultiRunConfig"):
        if len(self.initial_slider_poses) != len(other.initial_slider_poses):
            return False
        for i in range(len(self.initial_slider_poses)):
            if not self.initial_slider_poses[i] == other.initial_slider_poses[i]:
                return False
        if len(self.target_slider_poses) != len(other.target_slider_poses):
            return False
        for i in range(len(self.target_slider_poses)):
            if not self.target_slider_poses[i] == other.target_slider_poses[i]:
                return False

        return (
            self.num_runs == other.num_runs
            and self.seed == other.seed
            and self.max_attempt_duration == other.max_attempt_duration
            and self.trans_tol == other.trans_tol
            and self.rot_tol == other.rot_tol
            and self.evaluate_final_pusher_position
            == other.evaluate_final_pusher_position
            and self.evaluate_final_slider_rotation
            == other.evaluate_final_slider_rotation
        )


@dataclass
class PlanarPushingSimConfig:
    dynamics_config: SliderPusherSystemConfig
    slider: RigidBody
    contact_model: ContactModel = ContactModel.kHydroelastic
    visualize_desired: bool = False
    slider_goal_pose: Optional[PlanarPose] = None
    pusher_start_pose: PlanarPose = field(
        default_factory=lambda: PlanarPose(x=0.0, y=0.5, theta=0.0)
    )
    slider_start_pose: PlanarPose = field(
        default_factory=lambda: PlanarPose(x=0.0, y=0.5, theta=0.0)
    )
    default_joint_positions: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.array(
            [0.666, 1.039, -0.7714, -2.0497, 1.3031, 0.6729, -1.0252]
        )
    )
    time_step: float = 1e-3
    closed_loop: bool = True
    draw_frames: bool = False
    use_realtime: bool = False
    delay_before_execution: float = 5.0
    save_plots: bool = False
    mpc_config: HybridMpcConfig = field(default_factory=lambda: HybridMpcConfig())
    diffusion_policy_config: DiffusionPolicyConfig = None
    scene_directive_name: str = "planar_pushing_iiwa_plant_hydroelastic.yaml"
    use_hardware: bool = False
    pusher_z_offset: float = 0.05
    camera_configs: List[CameraConfig] = None
    domain_randomization: bool = False
    randomize_camera: bool = False
    log_dir: str = (
        None  # directory for logging rollouts from output_feedback_table_environments
    )
    multi_run_config: MultiRunConfig = None
    slider_physical_properties: PhysicalProperties = None

    @classmethod
    def from_traj(cls, trajectory: PlanarPushingTrajectory, **kwargs):
        return cls(
            dynamics_config=trajectory.config.dynamics_config,
            slider=trajectory.config.dynamics_config.slider,
            pusher_start_pose=trajectory.initial_pusher_planar_pose,
            slider_start_pose=trajectory.initial_slider_planar_pose,
            slider_goal_pose=trajectory.target_slider_planar_pose,
            **kwargs,
        )

    @classmethod
    def from_yaml(cls, cfg: OmegaConf):
        slider_physical_properties: PhysicalProperties = hydra.utils.instantiate(
            cfg.physical_properties
        )
        
        # Create sim_config with mandatory fields
        # TODO: read slider directly from yaml instead of if statement
        if cfg.slider_type == "box":
            slider: RigidBody = get_box(slider_physical_properties.mass)
        elif cfg.slider_type == "tee":
            slider: RigidBody = get_tee(slider_physical_properties.mass)
        elif cfg.slider_type == "arbitrary":
            slider = get_arbitrary(
                cfg.arbitrary_shape_pickle_path,
                slider_physical_properties.mass,
            )
        else:
            raise ValueError(f"Slider type not yet implemented: {cfg.slider_type}")        
        dynamics_config: SliderPusherSystemConfig = hydra.utils.instantiate(
            cfg.dynamics_config,
        )
        dynamics_config.slider = slider
        slider_goal_pose: PlanarPose = hydra.utils.instantiate(cfg.slider_goal_pose)
        pusher_start_pose: PlanarPose = hydra.utils.instantiate(cfg.pusher_start_pose)
        sim_config = cls(
            dynamics_config=dynamics_config,
            slider=slider,
            contact_model=eval(cfg.contact_model),
            visualize_desired=cfg.visualize_desired,
            slider_goal_pose=slider_goal_pose,
            pusher_start_pose=pusher_start_pose,
            time_step=cfg.time_step,
            closed_loop=cfg.closed_loop,
            draw_frames=cfg.draw_frames,
            use_realtime=cfg.use_realtime,
            delay_before_execution=cfg.delay_before_execution,
            save_plots=cfg.save_plots,
            scene_directive_name=cfg.scene_directive_name,
            use_hardware=cfg.use_hardware,
            pusher_z_offset=cfg.pusher_z_offset,
            log_dir=cfg.log_dir,
            domain_randomization=cfg.domain_randomization,
            randomize_camera=cfg.randomize_camera,
            slider_physical_properties=slider_physical_properties,
        )

        # Optional fields
        if "slider_start_pose" in cfg:
            sim_config.slider_start_pose = hydra.utils.instantiate(
                cfg.slider_start_pose
            )
        if "default_joint_positions" in cfg:
            sim_config.default_joint_positions = np.array(cfg.default_joint_positions)
        if "mpc_config" in cfg:
            sim_config.mpc_config = hydra.utils.instantiate(cfg.mpc_config)
        if "diffusion_policy_config" in cfg:
            sim_config.diffusion_policy_config = hydra.utils.instantiate(
                cfg.diffusion_policy_config
            )
        if "camera_configs" in cfg and cfg.camera_configs:
            camera_configs = []
            for camera_config in cfg.camera_configs:
                if camera_config.orientation == "default":
                    # default camera orientation is looking down
                    X_PB = Transform(
                        RigidTransform(
                            RotationMatrix.MakeXRotation(np.pi),
                            np.array(camera_config.position),
                        )
                    )
                else:
                    orientation = RollPitchYaw(
                        roll=camera_config.orientation.roll,
                        pitch=camera_config.orientation.pitch,
                        yaw=camera_config.orientation.yaw,
                    )

                    X_PB = Transform(
                        RigidTransform(orientation, np.array(camera_config.position))
                    )

                camera_configs.append(
                    CameraConfig(
                        name=camera_config.name,
                        X_PB=X_PB,
                        width=camera_config.width,
                        height=camera_config.height,
                        show_rgb=camera_config.show_rgb,
                        center_x=camera_config.center_x,
                        center_y=camera_config.center_y,
                        focal=CameraConfig.FocalLength(
                            x=camera_config.focal_x, y=camera_config.focal_y
                        ),
                        background=Rgba(
                            255.0 / 255.0, 228.0 / 255.0, 196.0 / 255.0, 1.0
                        ),
                    )
                )
            sim_config.camera_configs = camera_configs
        if "multi_run_config" in cfg and cfg.multi_run_config:
            sim_config.multi_run_config = hydra.utils.instantiate(cfg.multi_run_config)

        return sim_config

    def __eq__(self, other: "PlanarPushingSimConfig"):
        # Note: this function does not check equality for MPC config

        # Check camera configs
        if self.camera_configs is None and other.camera_configs is not None:
            return False
        if self.camera_configs is not None and other.camera_configs is None:
            return False
        if self.camera_configs is not None:
            for camera_config in self.camera_configs:
                if camera_config not in other.camera_configs:
                    return False

        return (
            self.slider == other.slider
            and self.dynamics_config == other.dynamics_config
            and self.contact_model == other.contact_model
            and self.visualize_desired == other.visualize_desired
            and self.slider_goal_pose == other.slider_goal_pose
            and self.pusher_start_pose == other.pusher_start_pose
            and self.time_step == other.time_step
            and self.closed_loop == other.closed_loop
            and self.draw_frames == other.draw_frames
            and self.use_realtime == other.use_realtime
            and self.delay_before_execution == other.delay_before_execution
            and self.save_plots == other.save_plots
            and self.scene_directive_name == other.scene_directive_name
            and self.use_hardware == other.use_hardware
            and self.pusher_z_offset == other.pusher_z_offset
            and self.log_dir == other.log_dir
            and np.allclose(self.default_joint_positions, other.default_joint_positions)
            and self.diffusion_policy_config == other.diffusion_policy_config
            and self.multi_run_config == other.multi_run_config,
            self.domain_randomization == other.domain_randomization,
            self.randomize_camera == other.randomize_camera,
        )
