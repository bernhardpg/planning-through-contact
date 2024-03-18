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
    RotationMatrix
)
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
    MultiRunConfig
)
from planning_through_contact.simulation.controllers.hybrid_mpc import HybridMpcConfig
from planning_through_contact.experiments.utils import (
    get_box,
    get_tee,
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
    scene_directive_name: str = "planar_pushing_iiwa_plant_hydroelastic.yaml"
    use_hardware: bool = False
    pusher_z_offset: float = 0.05
    camera_configs: List[CameraConfig] = None # TODO: make this a list of cameras
    collect_data: bool = False
    data_dir: str = None # remove this

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
        # TODO: read slider directly from yaml instead of if statement
        slider: RigidBody = get_box() if cfg.slider_type == "box" else get_tee()
        dynamics_config: SliderPusherSystemConfig = hydra.utils.instantiate(
            cfg.dynamics_config,
        )
        dynamics_config.slider = slider

        slider_goal_pose: PlanarPose = hydra.utils.instantiate(cfg.slider_goal_pose)
        pusher_start_pose: PlanarPose = hydra.utils.instantiate(cfg.pusher_start_pose)
        slider_start_pose: PlanarPose = hydra.utils.instantiate(cfg.slider_start_pose)
        default_joint_positions = np.array(cfg.default_joint_positions)
        mpc_config: HybridMpcConfig = hydra.utils.instantiate(cfg.mpc_config)
        breakpoint()
        
        if 'camera_configs' in cfg and cfg.camera_configs:
            for camera_config in cfg.camera_configs:
                if cfg.camera_config.orientation == 'default':
                    X_PB = Transform(
                        RigidTransform(
                            RotationMatrix.MakeXRotation(np.pi),
                            np.array([0.5, 0.0, 1.0])
                        )
                    )
                else:
                    # TODO: X_PB from yaml
                    raise NotImplementedError
                camera_config = CameraConfig(
                    name=cfg.camera_config.name,
                    X_PB=X_PB,
                    width=cfg.camera_config.width,
                    height=cfg.camera_config.height,
                    show_rgb=cfg.camera_config.show_rgb,
                )
        else:
            camera_config = None
        
        if 'multi_run_config' in cfg and cfg.multi_run_config:
            multi_run_config = hydra.utils.instantiate(cfg.multi_run_config)
        else:
            multi_run_config = None

        return cls(
            dynamics_config=dynamics_config,
            slider=slider,
            contact_model=eval(cfg.contact_model),
            visualize_desired=cfg.visualize_desired,
            slider_goal_pose=slider_goal_pose,
            pusher_start_pose=pusher_start_pose,
            slider_start_pose=slider_start_pose,
            default_joint_positions=default_joint_positions,
            time_step=cfg.time_step,
            closed_loop=cfg.closed_loop,
            draw_frames=cfg.draw_frames,
            use_realtime=cfg.use_realtime,
            delay_before_execution=cfg.delay_before_execution,
            save_plots=cfg.save_plots,
            mpc_config=mpc_config,
            scene_directive_name=cfg.scene_directive_name,
            use_hardware=cfg.use_hardware,
            pusher_z_offset=cfg.pusher_z_offset,
            camera_config=camera_config,
            collect_data=cfg.collect_data,
            data_dir=cfg.data_dir,
            multi_run_config=multi_run_config
        )