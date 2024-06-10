import importlib
import logging
import os
import pathlib

import hydra
import numpy as np
from omegaconf import OmegaConf
from pydrake.all import StartMeshcat

from planning_through_contact.simulation.controllers.diffusion_policy_source import (
    DiffusionPolicySource,
)
from planning_through_contact.simulation.controllers.robot_system_base import (
    RobotSystemBase,
)
from planning_through_contact.simulation.environments.output_feedback_table_environment import (
    OutputFeedbackTableEnvironment,
)
from planning_through_contact.simulation.planar_pushing.planar_pushing_sim_config import (
    PlanarPushingSimConfig,
)
from planning_through_contact.simulation.sim_utils import (
    get_slider_sdf_path,
    models_folder,
)
from planning_through_contact.tools.utils import (
    create_processed_mesh_primitive_sdf_file,
    load_primitive_info,
)


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parents[3].joinpath("config", "sim_config")),
)
def run_sim(cfg: OmegaConf):
    logging.basicConfig(level=logging.INFO)

    # start meshcat
    print(f"station meshcat")
    station_meshcat = StartMeshcat()

    # load sim_config
    sim_config = PlanarPushingSimConfig.from_yaml(cfg)
    print(f"Initial finger pose: {sim_config.pusher_start_pose}")
    print(f"Target slider pose: {sim_config.slider_goal_pose}")

    if cfg.slider_type == "arbitrary":
        # create arbitrary shape sdf file
        create_arbitrary_shape_sdf_file(cfg, sim_config)

    # Diffusion Policy source
    position_source = DiffusionPolicySource(sim_config.diffusion_policy_config)

    # Set up position controller
    # TODO: load with hydra instead (currently giving camera config errors)
    module_name, class_name = cfg.robot_station._target_.rsplit(".", 1)
    robot_system_class = getattr(importlib.import_module(module_name), class_name)
    position_controller: RobotSystemBase = robot_system_class(
        sim_config=sim_config, meshcat=station_meshcat
    )

    # Set up environment
    environment = OutputFeedbackTableEnvironment(
        desired_position_source=position_source,
        robot_system=position_controller,
        sim_config=sim_config,
        station_meshcat=station_meshcat,
        arbitrary_shape_pickle_path=cfg.arbitrary_shape_pickle_path,
    )

    # Configure sim and recording
    recording_name = "diffusion_policy_roll_out.html"
    # environment.export_diagram("diffusion_environment_diagram.pdf")
    if sim_config.multi_run_config is None:
        end_time = 100.0
        seed = "N/A (no multi_run_config seed provided)"
    else:
        num_runs = sim_config.multi_run_config.num_runs
        max_attempt_duration = sim_config.multi_run_config.max_attempt_duration
        seed = sim_config.multi_run_config.seed
        end_time = num_runs * max_attempt_duration

    successful_idx, save_dir = environment.simulate(
        end_time, recording_file=recording_name
    )

    # Update logs and save config file
    OmegaConf.save(cfg, f"{save_dir}/sim_config.yaml")
    with open(f"{cfg.log_dir}/checkpoint_statistics.txt", "a") as f:
        f.write(f"{sim_config.diffusion_policy_config.checkpoint}\n")
        f.write(f"Seed: {seed}\n")
        f.write(
            f"Success ratio: {len(successful_idx)} / {num_runs} = {100.0*len(successful_idx) / num_runs:.3f}%\n"
        )
        f.write(f"Success_idx: {successful_idx}\n")
        f.write(f"Save dir: {save_dir}\n")
        f.write("\n")

    if cfg.slider_type == "arbitrary":
        # Remove the sdf file.
        sdf_path = get_slider_sdf_path(sim_config, models_folder)
        if os.path.exists(sdf_path):
            os.remove(sdf_path)


# TODO: This is duplicated from scripts/planar_pushing/run_data_generation.py. Move this
# to a common location.
def create_arbitrary_shape_sdf_file(cfg: OmegaConf, sim_config: PlanarPushingSimConfig):
    sdf_path = get_slider_sdf_path(sim_config, models_folder)
    if os.path.exists(sdf_path):
        os.remove(sdf_path)

    translation = -np.concatenate(
        [sim_config.slider.geometry.com_offset.flatten(), [0]]
    )  # Plan assumes that object frame = CoM frame

    primitive_info = load_primitive_info(cfg.arbitrary_shape_pickle_path)
    create_processed_mesh_primitive_sdf_file(
        primitive_info=primitive_info,
        visual_mesh_file_path=cfg.arbitrary_shape_visual_mesh_path,
        physical_properties=hydra.utils.instantiate(cfg.physical_properties),
        global_translation=translation,
        output_file_path=sdf_path,
        model_name="arbitrary",
        base_link_name="arbitrary",
        is_hydroelastic="hydroelastic" in cfg.contact_model.lower(),
        rgba=[0.0, 0.0, 0.0, 1.0],
        com_override=[0.0, 0.0, 0.0],  # Plan assumes that object frame = CoM frame
    )


if __name__ == "__main__":
    """
    Configure sim config through hydra yaml file
    Ex: python scripts/diffusion_policy/planar_pushing/run_sim_diffusion.py --config-name=actuated_cylinder_sim_config.yaml
    """
    run_sim()
