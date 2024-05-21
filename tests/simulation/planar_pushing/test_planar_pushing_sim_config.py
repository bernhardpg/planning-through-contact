from dataclasses import dataclass, field
from typing import List, Optional

import hydra
import numpy as np
import numpy.typing as npt
import pytest
from omegaconf import OmegaConf
from pydrake.all import RollPitchYaw
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
from planning_through_contact.simulation.planar_pushing.planar_pushing_sim_config import (
    MultiRunConfig,
    PlanarPushingSimConfig,
)
from planning_through_contact.simulation.sim_utils import get_slider_start_poses


@pytest.fixture
def multi_run_config() -> MultiRunConfig:
    return MultiRunConfig(
        num_runs=1,
        max_attempt_duration=100,
        seed=163,
        slider_type="box",
        pusher_start_pose=PlanarPose(x=0.5, y=0.25, theta=0.0),
        slider_goal_pose=PlanarPose(x=0.5, y=0.0, theta=0.0),
        workspace_width=0.5,
        workspace_height=0.5,
    )


@pytest.fixture()
def diffusion_policy_config() -> DiffusionPolicyConfig:
    return DiffusionPolicyConfig(
        checkpoint="checkpoint",
        initial_pusher_pose=PlanarPose(x=0.5, y=0.25, theta=0.0),
        target_slider_pose=PlanarPose(x=0.5, y=0.0, theta=0.0),
    )


@pytest.fixture()
def sim_config() -> PlanarPushingSimConfig:
    slider = get_box()
    dynamics_config = SliderPusherSystemConfig(
        pusher_radius=0.015,
        friction_coeff_table_slider=0.5,
        friction_coeff_slider_pusher=0.1,
        grav_acc=9.81,
        integration_constant=0.3,
        force_scale=0.01,
    )
    dynamics_config.slider = slider

    return PlanarPushingSimConfig(
        dynamics_config=dynamics_config,
        slider=slider,
        contact_model=ContactModel.kHydroelastic,
        visualize_desired=True,
        slider_goal_pose=PlanarPose(x=0.5, y=0.0, theta=0.0),
        pusher_start_pose=PlanarPose(x=0.5, y=0.25, theta=0.0),
        default_joint_positions=np.array(
            [0.0776, 1.0562, 0.3326, -1.3048, 2.7515, -0.8441, 0.5127]
        ),
        time_step=0.001,
        closed_loop=False,
        draw_frames=True,
        use_realtime=False,
        delay_before_execution=5.0,
        save_plots=False,
        diffusion_policy_config=DiffusionPolicyConfig(
            checkpoint="checkpoint",
            initial_pusher_pose=PlanarPose(x=0.5, y=0.25, theta=0.0),
            target_slider_pose=PlanarPose(x=0.5, y=0.0, theta=0.0),
            cfg_overrides={"n_actions": 8},
        ),
        use_hardware=False,
        pusher_z_offset=0.03,
        camera_configs=None,
        log_dir="diffusion_policy_logs",
        multi_run_config=MultiRunConfig(
            num_runs=1,
            max_attempt_duration=100,
            seed=163,
            slider_type="box",
            pusher_start_pose=PlanarPose(x=0.5, y=0.25, theta=0.0),
            slider_goal_pose=PlanarPose(x=0.5, y=0.0, theta=0.0),
            workspace_width=0.5,
            workspace_height=0.5,
        ),
        scene_directive_name="planar_pushing_iiwa_plant_hydroelastic.yaml",
    )


def test_multi_run_config_equality(multi_run_config):
    assert multi_run_config == multi_run_config


def test_from_yaml(sim_config):
    sim_config_from_yaml = PlanarPushingSimConfig.from_yaml(
        OmegaConf.load("tests/simulation/planar_pushing/test_sim_config.yaml")
    )
    assert sim_config == sim_config_from_yaml
