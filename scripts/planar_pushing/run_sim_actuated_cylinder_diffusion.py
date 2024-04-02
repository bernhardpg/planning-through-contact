import numpy as np
import argparse
import os
from tqdm import tqdm
from typing import List, Tuple, Optional
from omegaconf import OmegaConf

from pydrake.all import ContactModel, StartMeshcat
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

import logging

from planning_through_contact.geometry.collision_geometry.box_2d import Box2d
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.planar.planar_pushing_trajectory import (
    PlanarPushingTrajectory,
)
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

from planning_through_contact.planning.planar.planar_plan_config import (
    MultiRunConfig
)

from planning_through_contact.planning.planar.planar_plan_config import (
    BoxWorkspace,
    PlanarPlanConfig,
    PlanarPushingStartAndGoal,
    PlanarPushingWorkspace,
    MultiRunConfig
)

from planning_through_contact.geometry.planar.non_collision import (
    check_finger_pose_in_contact_location,
)

from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    CollisionGeometry,
    ContactLocation,
    PolytopeContactLocation,
)

from planning_through_contact.experiments.utils import (
    get_default_plan_config,
)


def run_sim(
    plan: str, # TODO: remove the need for this argument
    checkpoint: str,
    num_runs: int,
    max_attempt_duration: float,
    seed: int,
    traj_idx: int,
    data_collection_dir: str = None,
    save_recording: bool = False,
    debug: bool = False,
    station_meshcat=None,
):
    logging.basicConfig(level=logging.INFO)
    logging.getLogger(
        "planning_through_contact.simulation.planar_pushing.pusher_pose_controller"
    ).setLevel(logging.DEBUG)

    # load sim_config
    cfg = OmegaConf.load('config/sim_config/test_sim_config.yaml')
    sim_config = PlanarPushingSimConfig.from_yaml(cfg)
    sim_config.multi_run_config = get_multi_run_config(
        num_runs,
        max_attempt_duration,
        seed=seed,
        slider_type=cfg.slider_type,
        traj_idx=traj_idx,
        initial_pusher_pose=sim_config.pusher_start_pose,
        target_slider_pose=sim_config.slider_goal_pose
    )

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
        if num_runs == 1 and traj_idx is not None:
            f.write(f"Trajectory idx: {traj_idx}\n")
        f.write(f"Success ratio: {len(successful_idx)} / {num_runs} = {100.0*len(successful_idx) / num_runs:.3f}%\n")
        f.write(f"Success_idx: {successful_idx}\n")
        f.write(f"Save dir: {save_dir}\n")
        f.write("\n")

def run_multiple(
    plans: list,
    save_dir: str,
    station_meshcat=None, 
):
    print(f"Running {len(plans)} plans\n{plans}")
    for plan in tqdm(plans):
        if not os.path.exists(plan):
            print(f"Plan {plan} does not exist. Skipping.")
            continue
        run_sim(
            plan,
            data_collection_dir=save_dir,
            save_recording=False,
            debug=False,
            station_meshcat=station_meshcat,
        )
        station_meshcat.Delete()
        station_meshcat.DeleteAddedControls()

# similar to the functions in scripts/planar_pushing/create_plan.py
        
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

def get_slider_start_poses(
    seed: int,
    num_plans: int,
    workspace: PlanarPushingWorkspace,
    config: PlanarPlanConfig,
    pusher_pose: PlanarPose,
    limit_rotations: bool = True,  # Use this to start with
) -> List[PlanarPushingStartAndGoal]:
    # We want the plans to always be the same
    np.random.seed(seed)
    slider = config.slider_geometry
    slider_initial_poses = []
    for _ in range(num_plans):
        slider_initial_pose = _get_slider_pose_within_workspace(
            workspace, slider, pusher_pose, config, limit_rotations
        )
        slider_initial_poses.append(slider_initial_pose)

    return slider_initial_poses

def get_multi_run_config(num_runs, 
                         max_attempt_duration, 
                         seed, 
                         slider_type='tee',
                         traj_idx = None,
                         initial_pusher_pose=PlanarPose(0.5, 0.25, 0.0),
                         target_slider_pose=PlanarPose(0.5, 0.0, 0.0)):
    # Set up multi run config
    config = get_default_plan_config(
        slider_type=slider_type,
        pusher_radius=0.015,
        hardware=False,
    )
    # update config
    config.contact_config.lam_min = 0.15
    config.contact_config.lam_max = 0.85
    config.non_collision_cost.distance_to_object_socp = 0.25   

    workspace = PlanarPushingWorkspace(
        slider=BoxWorkspace(
            width=0.35,
            height=0.5,
            center=np.array([target_slider_pose.x, target_slider_pose.y]),
            buffer=0,
        ),
    )
    initial_slider_poses = get_slider_start_poses(
        seed=seed,
        num_plans=max(num_runs, 100),
        workspace=workspace,
        config=config,
        pusher_pose=initial_pusher_pose,
        limit_rotations=False,
    )

    if num_runs == 1:
        if traj_idx is not None:
            initial_slider_poses = [initial_slider_poses[traj_idx]]
        else:
            initial_slider_poses = [initial_slider_poses[0]]
    else:
        initial_slider_poses = initial_slider_poses[:num_runs]

    # For push_tee_v2 demo, hard coding the initial slider poses
    # successful_idx = [0, 3, 9, 30, 0, 53, 65, 53, 54, 51]
    # initial_slider_poses = [initial_slider_poses[i] for i in successful_idx]
    # initial_slider_poses[4] = PlanarPose(0.513585856901175, -0.0404027427983526, 1.160064053488128)

    return MultiRunConfig(
        initial_slider_poses=initial_slider_poses,
        target_slider_poses=[target_slider_pose for _ in range(len(initial_slider_poses))],
        max_attempt_duration=max_attempt_duration
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--num_runs", type=int, default=1, help="Number of runs to simulate")
    parser.add_argument("--max_attempt_duration", type=float, default=90.0, help="Max duration for each run")
    parser.add_argument("--seed", type=int, default=163, help="Seed for random number generator")
    parser.add_argument("--traj_idx", type=int, default=None, help="Index of trajectory to use")
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
        plan=plan,
        checkpoint=checkpoint,
        num_runs=args.num_runs,
        max_attempt_duration=args.max_attempt_duration,
        seed=args.seed,
        traj_idx=args.traj_idx,
        data_collection_dir=None,
        save_recording=True,
        debug=False,
        station_meshcat=station_meshcat,
    )