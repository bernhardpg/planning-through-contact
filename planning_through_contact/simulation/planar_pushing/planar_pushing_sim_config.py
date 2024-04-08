from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np
import numpy.typing as npt

import hydra
import torch
from omegaconf import OmegaConf

from pydrake.multibody.plant import (
    ContactModel,
)
from pydrake.systems.sensors import (
    CameraConfig
)
from pydrake.math import (
    RigidTransform, 
    RotationMatrix,
)
from pydrake.all import RollPitchYaw
from pydrake.common.schema import (
    Transform
)

from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.planar.planar_pushing_trajectory import (
    PlanarPushingTrajectory,
)
from planning_through_contact.geometry.rigid_body import RigidBody
from planning_through_contact.planning.planar.planar_plan_config import (
    SliderPusherSystemConfig,
    BoxWorkspace,
    PlanarPushingWorkspace,
)
from planning_through_contact.simulation.controllers.hybrid_mpc import HybridMpcConfig
from planning_through_contact.simulation.controllers.diffusion_policy_source import DiffusionPolicyConfig
from planning_through_contact.experiments.utils import (
    get_box,
    get_tee,
    get_default_plan_config,
    get_slider_start_poses,
)

class MultiRunConfig:
    def __init__(
            self,
            num_runs, 
            max_attempt_duration, 
            seed, 
            slider_type,
            pusher_start_pose,
            slider_goal_pose,
            workspace_width=0.35,
            workspace_height=0.5,
    ):
        # Set up multi run config
        config = get_default_plan_config(
            slider_type=slider_type,
            pusher_radius=0.015,
            hardware=False,
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
    
    def __str__(self):
        slider_pose_str = f"initial_slider_poses: {self.initial_slider_poses}"
        target_pose_str = f"target_slider_poses: {self.target_slider_poses}"
        return f"{slider_pose_str}\n{target_pose_str}\nmax_attempt_duration: {self.max_attempt_duration}"


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
    collect_data: bool = False
    data_collection_dir: str = None # directory for data collection
    log_dir: str = None # directory for logging rollouts from output_feedback_table_environment

    multi_run_config: MultiRunConfig = None

    @classmethod
    def from_traj(cls, trajectory: PlanarPushingTrajectory, **kwargs):
        return cls(
            dynamics_config=trajectory.config.dynamics_config,
            slider=trajectory.config.dynamics_config.slider,
            pusher_start_pose=trajectory.initial_pusher_planar_pose,
            slider_start_pose=trajectory.initial_slider_planar_pose,
            slider_goal_pose=trajectory.target_slider_planar_pose,
            **kwargs
        )

    @classmethod
    def from_yaml(cls, cfg: OmegaConf):
        # Create sim_config with mandatory fields
        # TODO: read slider directly from yaml instead of if statement
        slider: RigidBody = get_box() if cfg.slider_type == "box" else get_tee()
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
            collect_data=cfg.collect_data,
            data_collection_dir=cfg.data_collection_dir,
            log_dir=cfg.log_dir,
        )

        # Optional fields
        if 'slider_start_pose' in cfg:
            sim_config.slider_start_pose = hydra.utils.instantiate(cfg.slider_start_pose)
        if 'default_joint_positions' in cfg:
            sim_config.default_joint_positions = np.array(cfg.default_joint_positions)
        if 'mpc_config' in cfg:
            sim_config.mpc_config = hydra.utils.instantiate(cfg.mpc_config)
        if 'diffusion_policy_config' in cfg:
            sim_config.diffusion_policy_config = hydra.utils.instantiate(cfg.diffusion_policy_config)
        if 'camera_configs' in cfg and cfg.camera_configs:
            camera_configs = []
            for camera_config in cfg.camera_configs:
                if camera_config.orientation == 'default':
                    X_PB = Transform(
                        RigidTransform(
                            RotationMatrix.MakeXRotation(np.pi),
                            np.array(camera_config.position)
                        )
                    )
                else:
                    orientation = RollPitchYaw(
                        roll=camera_config.orientation.roll,
                        pitch=camera_config.orientation.pitch,
                        yaw=camera_config.orientation.yaw
                    )
                    X_PB=Transform(
                        RigidTransform(orientation, np.array(camera_config.position))
                    )
                
                camera_configs.append(
                    CameraConfig(
                        name=camera_config.name,
                        X_PB=X_PB,
                        width=camera_config.width,
                        height=camera_config.height,
                        show_rgb=camera_config.show_rgb,
                    )
                )
            sim_config.camera_configs = camera_configs
        if 'multi_run_config' in cfg and cfg.multi_run_config:
            sim_config.multi_run_config = hydra.utils.instantiate(cfg.multi_run_config)

        return sim_config