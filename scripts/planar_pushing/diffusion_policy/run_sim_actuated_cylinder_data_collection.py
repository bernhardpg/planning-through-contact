import os
from tqdm import tqdm
import hydra
from omegaconf import OmegaConf
import pathlib
import importlib
import logging

from pydrake.all import StartMeshcat

from planning_through_contact.geometry.planar.planar_pushing_trajectory import (
    PlanarPushingTrajectory,
)
from planning_through_contact.simulation.controllers.replay_position_source import (
    ReplayPositionSource,
)
from planning_through_contact.simulation.environments.data_collection_table_environment import (
    DataCollectionTableEnvironment,
)
from planning_through_contact.simulation.planar_pushing.planar_pushing_sim_config import (
    PlanarPushingSimConfig,
)
from planning_through_contact.simulation.controllers.robot_system_base import (
    RobotSystemBase,
)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parents[3].joinpath(
        'config','sim_config'))
)
def run_sim(cfg: OmegaConf):
    logging.basicConfig(level=logging.INFO)
    logging.getLogger(
        "planning_through_contact.simulation.planar_pushing.pusher_pose_controller"
    ).setLevel(logging.DEBUG)

    # start meshcat
    print(f"station meshcat")
    station_meshcat = StartMeshcat()

    # load sim_config
    sim_config = PlanarPushingSimConfig.from_yaml(cfg)
    print(f"Initial finger pose: {sim_config.pusher_start_pose}")
    print(f"Target slider pose: {sim_config.slider_goal_pose}")
    
    # TODO: move into data_collection config
    plan="trajectories/data_collection_trajectories_box_v2/run_0/traj_0/trajectory/traj_rounded.pkl"
    traj = PlanarPushingTrajectory.load(plan)
    position_source = ReplayPositionSource(
        traj=traj,
        dt = 0.025,
        delay=sim_config.delay_before_execution
    )

    ## Set up position controller
    # TODO: load with hydra instead (currently giving camera config errors)
    # overrides = {'sim_config': sim_config, 'meshcat': station_meshcat}
    # position_controller: RobotSystemBase = hydra.utils.instantiate(cfg.robot_station, **overrides)
    
    module_name, class_name = cfg.robot_station._target_.rsplit(".", 1)
    robot_system_class = getattr(importlib.import_module(module_name), class_name)
    position_controller: RobotSystemBase = robot_system_class(
        sim_config=sim_config, 
        meshcat=station_meshcat
    )

    environment = DataCollectionTableEnvironment(
        desired_position_source=position_source,
        robot_system=position_controller,
        sim_config=sim_config,
        state_estimator_meshcat=station_meshcat,
    )

    # TODO: move into data_collection config
    save_recording = False
    recording_name = (
        "recording.html"
        if save_recording
        else None
    )
    environment.export_diagram("environment_diagram.pdf")
    environment.simulate(traj.end_time + 0.5, recording_file=recording_name)

# TODO: fix this
# @hydra.main(
#     version_base=None,
#     config_path=str(pathlib.Path(__file__).parents[3].joinpath(
#         'config','sim_config'))
# )
# def run_multiple(cfg: OmegaConf):
#     print(f"Running {len(plans)} plans\n{plans}")
#     for plan in tqdm(plans):
#         if not os.path.exists(plan):
#             print(f"Plan {plan} does not exist. Skipping.")
#             continue
#         run_sim(
#             plan,
#             data_collection_dir=save_dir,
#             save_recording=False,
#             debug=False,
#             station_meshcat=station_meshcat,
#             state_estimator_meshcat=state_estimator_meshcat,
#         )
#         station_meshcat.Delete()
#         station_meshcat.DeleteAddedControls()
#         state_estimator_meshcat.Delete()


if __name__ == "__main__":
    run_sim()