from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
from pydrake.multibody.plant import (
    ContactModel,
)
from pydrake.systems.sensors import (
    CameraConfig
)

from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.planar.planar_pushing_trajectory import (
    PlanarPushingTrajectory,
)
from planning_through_contact.geometry.rigid_body import RigidBody
from planning_through_contact.planning.planar.planar_plan_config import (
    SliderPusherSystemConfig,
)
from planning_through_contact.simulation.controllers.hybrid_mpc import HybridMpcConfig


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
    camera_config: CameraConfig = None

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
