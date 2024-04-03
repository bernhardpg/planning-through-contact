import argparse
import logging

import hydra
import pathlib
from omegaconf import OmegaConf

from pydrake.all import StartMeshcat

from planning_through_contact.geometry.planar.planar_pose import PlanarPose

from planning_through_contact.simulation.controllers.cylinder_actuated_station import (
    CylinderActuatedStation,
)
from planning_through_contact.simulation.controllers.diffusion_policy_source import (
    DiffusionPolicySource,
)

from planning_through_contact.simulation.environments.output_feedback_table_environment import (
    OutputFeedbackTableEnvironment,
)
from planning_through_contact.simulation.planar_pushing.planar_pushing_sim_config import (
    PlanarPushingSimConfig,
)

def run_sim(
    # cfg: OmegaConf,
    checkpoint: str,
    num_runs: int,
    max_attempt_duration: float,
    seed: int,
    save_recording: bool = False,
    station_meshcat=None,
):
    logging.basicConfig(level=logging.INFO)
    logging.getLogger(
        "planning_through_contact.simulation.planar_pushing.pusher_pose_controller"
    ).setLevel(logging.DEBUG)

    # load sim_config
    cfg = OmegaConf.load('config/sim_config/actuated_cylinder_sim_config.yaml')
    sim_config = PlanarPushingSimConfig.from_yaml(cfg)

    # print some debugging config info
    print(f"Initial finger pose: {sim_config.pusher_start_pose}")
    print(f"Target slider pose: {sim_config.slider_goal_pose}")

    # Diffusion Policy source
    position_source = DiffusionPolicySource(sim_config=sim_config, checkpoint=checkpoint)

    ## Set up position controller
    position_controller = CylinderActuatedStation(
        sim_config=sim_config, meshcat=station_meshcat
    )

    environment = OutputFeedbackTableEnvironment(
        desired_position_source=position_source,
        robot_system=position_controller,
        sim_config=sim_config,
        station_meshcat=station_meshcat,
    )
    recording_name = (
        "diffusion_policy_roll_out.html"
        if save_recording
        else None
    )
    environment.export_diagram("diffusion_environment_diagram.pdf")
    end_time = max(100.0, num_runs * max_attempt_duration)
    successful_idx, save_dir = environment.simulate(end_time, recording_file=recording_name)
    # if num_runs > 1:
    with open("diffusion_policy_logs/checkpoint_statistics.txt", "a") as f:
        f.write(f"{checkpoint}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Success ratio: {len(successful_idx)} / {num_runs} = {100.0*len(successful_idx) / num_runs:.3f}%\n")
        f.write(f"Success_idx: {successful_idx}\n")
        f.write(f"Save dir: {save_dir}\n")
        f.write("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--num-runs", type=int, default=1, help="Number of runs to simulate")
    parser.add_argument("--max-attempt-duration", type=float, default=90.0, help="Max duration for each run")
    parser.add_argument("--seed", type=int, default=163, help="Seed for random number generator")
    args = parser.parse_args()

    if args.checkpoint is None:
        # checkpoint='/home/adam/workspace/gcs-diffusion/data/outputs/push_tee_v1_sc/checkpoints/epoch_148.ckpt'
        # checkpoint='/home/adam/workspace/gcs-diffusion/data/outputs/push_tee_v2/checkpoints/working_better.ckpt'
        checkpoint='/home/adam/workspace/gcs-diffusion/data/outputs/push_box_v2/checkpoints/epoch=0190-val_loss=0.005059.ckpt'
        # checkpoint='/home/adam/workspace/gcs-diffusion/data/outputs/push_tee_v2/checkpoints/epoch=0695-val_loss=0.035931.ckpt'
    else:
        checkpoint = args.checkpoint
    
    print(f"station meshcat")
    station_meshcat = StartMeshcat()
    # plan path is used to extract sim_config
    # the trajectory in plan path is not used
    plan = "trajectories/data_collection_trajectories_box_v2/run_0/traj_0/trajectory/traj_rounded.pkl"
    run_sim(
        checkpoint=checkpoint,
        num_runs=args.num_runs,
        max_attempt_duration=args.max_attempt_duration,
        seed=args.seed,
        save_recording=True,
        station_meshcat=station_meshcat,
    )