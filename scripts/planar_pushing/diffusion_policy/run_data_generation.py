import numpy as np
import math
import matplotlib.pyplot as plt
import pathlib
import os
import shutil
from tqdm import tqdm
import logging
from typing import List, Optional, Tuple
import pickle
import zarr
from PIL import Image
import importlib
import hydra
from omegaconf import OmegaConf

from pydrake.all import (
    Meshcat,
    StartMeshcat,
)

from planning_through_contact.simulation.planar_pushing.planar_pushing_sim_config import (
    PlanarPushingSimConfig,
)
from planning_through_contact.simulation.environments.data_collection_table_environment import (
    DataCollectionTableEnvironment,
    DataCollectionConfig,
)
from planning_through_contact.experiments.utils import (
    get_default_plan_config,
    get_default_solver_params,
)
from planning_through_contact.planning.planar.planar_plan_config import (
    BoxWorkspace,
    PlanarPlanConfig,
    PlanarPushingStartAndGoal,
    PlanarSolverParams,
    PlanarPushingWorkspace,
)
from planning_through_contact.planning.planar.planar_pushing_planner import (
    PlanarPushingPlanner,
)
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.simulation.sim_utils import get_slider_pose_within_workspace
from planning_through_contact.visualize.colors import COLORS
from planning_through_contact.visualize.planar_pushing import make_traj_figure
from planning_through_contact.geometry.planar.planar_pushing_trajectory import (
    PlanarPushingTrajectory,
)
from planning_through_contact.simulation.controllers.replay_position_source import (
    ReplayPositionSource,
)
from planning_through_contact.simulation.controllers.robot_system_base import (
    RobotSystemBase,
)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parents[3].joinpath(
        'config','sim_config'))
)
def main(cfg: OmegaConf):
    """
    Performs the data collection. Configure the data collection steps in the config file.
    The available steps are:
    - Generate plans
    - Render plans
    - Convert rendered plans to zarr format
    """

    ## Configure logs
    logging.getLogger(
        "planning_through_contact.simulation.environments.data_collection_table_environment"
    ).setLevel(logging.WARNING)
    logging.getLogger('drake').setLevel(logging.WARNING)

    ## Parse Configs
    sim_config: PlanarPushingSimConfig = PlanarPushingSimConfig.from_yaml(cfg)
    _print_sim_config_info(sim_config)

    data_collection_config: DataCollectionConfig = hydra.utils.instantiate(cfg.data_collection_config)
    _print_data_collection_config_info(data_collection_config)

    ## Generate plans
    if data_collection_config.generate_plans:
        generate_plans(data_collection_config)
        save_omegaconf(cfg, data_collection_config.plans_dir, config_name="config.yaml")

    ## Render plans
    if data_collection_config.render_plans:
        render_plans(sim_config, data_collection_config, cfg)
        save_omegaconf(cfg, data_collection_config.rendered_plans_dir, config_name="config.yaml")

    ## Convert data to zarr
    if data_collection_config.convert_to_zarr or data_collection_config.convert_to_zarr_reduce:
        convert_to_zarr(data_collection_config, debug=False)

def save_omegaconf(cfg: OmegaConf, dir: str, config_name: str="config.yaml"):
    with open(f"{dir}/{config_name}", "w") as f:
        OmegaConf.save(cfg, f)

def generate_plans(data_collection_config: DataCollectionConfig):
    """Generates plans according to the data collection config."""
    
    print("\nGenerating plans...")

    _create_directory(data_collection_config.plans_dir)

    ## Set up configs
    _plan_config = data_collection_config.plan_config
    config = get_default_plan_config(
        slider_type=_plan_config.slider_type,
        pusher_radius=_plan_config.pusher_radius,
        hardware=False,
    )
    solver_params = get_default_solver_params(debug=False, clarabel=False)
    config.contact_config.lam_min = _plan_config.contact_lam_min
    config.contact_config.lam_max = _plan_config.contact_lam_max
    config.non_collision_cost.distance_to_object_socp = _plan_config.distance_to_object_socp

    ## Set up workspace
    workspace = PlanarPushingWorkspace(
        slider=BoxWorkspace(
            width=_plan_config.width,
            height=_plan_config.height,
            center=np.array(_plan_config.center),
            buffer=_plan_config.buffer,
        ),
    )
    
    ## Get starts and goals
    plan_starts_and_goals = _get_plan_start_and_goals_to_point(
        seed=_plan_config.seed,
        num_plans=int(1.1*_plan_config.num_plans), # Add extra plans in case some fail
        workspace=workspace,
        config=config,
        point=_plan_config.center,
        init_pusher_pose=_plan_config.pusher_start_pose,
        limit_rotations=False,
        noise_final_pose=False,
    )
    print(f"Finished generating start and goal pairs.")

    ## Generate plans
    pbar = tqdm(total=_plan_config.num_plans)
    plan_idx = 0
    while plan_idx < _plan_config.num_plans and plan_idx < len(plan_starts_and_goals):
        plan = plan_starts_and_goals[plan_idx]
        success = create_multimodal_plans(
            plan_spec = plan,
            config=config,
            solver_params=solver_params,
            output_dir=data_collection_config.plans_dir,
            traj_name=f"traj_{plan_idx}",
            do_rounding=True,
            save_traj=True,
        )

        if success:
            plan_idx += 1
            pbar.update(1)
    print(f"Finished generating {plan_idx} plans.")
    if plan_idx < _plan_config.num_plans:
        print(f"Failed to generate all plans since the solver can fail.")

def create_plan(
    plan_spec: PlanarPushingStartAndGoal,
    config: PlanarPlanConfig,
    solver_params: PlanarSolverParams,
    output_dir: str = "",
    traj_name: str = "Untitled_traj",
    do_rounding: bool = True,
    save_traj: bool = False
) -> bool:
    """
    Create plans according to plan_spec and other config params.
    This function is largely inspired by the 'create_plan' function in
    'scripts/planar_pushing/create_plan.py'
    """

    # Set up folders
    folder_name = f"{output_dir}/{traj_name}"
    os.makedirs(folder_name, exist_ok=True)
    trajectory_folder = f"{folder_name}/trajectory"
    os.makedirs(trajectory_folder, exist_ok=True)
    analysis_folder = f"{folder_name}/analysis"
    os.makedirs(analysis_folder, exist_ok=True)


    planner = PlanarPushingPlanner(config)
    planner.config.start_and_goal = plan_spec
    planner.formulate_problem()
    path = planner.plan_path(solver_params)

    # We may get infeasible
    if path is not None:
        traj_relaxed = path.to_traj()
        traj_rounded = path.to_traj(rounded=True) if do_rounding else None

        if save_traj:
            if traj_rounded:
                traj_rounded.save(f"{trajectory_folder}/traj_rounded.pkl")
            else:
                traj_relaxed.save(f"{trajectory_folder}/traj_relaxed.pkl")  # type: ignore

        slider_color = COLORS["aquamarine4"].diffuse()

        if traj_rounded is not None:
            make_traj_figure(
                traj_rounded,
                filename=f"{analysis_folder}/rounded_traj",
                slider_color=slider_color,
                split_on_mode_type=True,
                show_workspace=False,
            )
    return path is not None

def create_multimodal_plans(
    plan_spec: PlanarPushingStartAndGoal,
    config: PlanarPlanConfig,
    solver_params: PlanarSolverParams,
    output_dir: str = "",
    traj_name: str = "Untitled_traj",
    do_rounding: bool = True,
    save_traj: bool = False
) -> bool:
    """
    Create plans according to plan_spec and other config params.
    This function is largely inspired by the 'create_plan' function in
    'scripts/planar_pushing/create_plan.py'
    """

    planner = PlanarPushingPlanner(config)
    planner.config.start_and_goal = plan_spec
    planner.formulate_problem()
    paths = planner.plan_multiple_paths(solver_params)

    if paths is None:
        return False
    
    for i in range(5):
        path = paths[i]

        # Set up folders
        folder_name = f"{output_dir}/{traj_name}_{i}"
        os.makedirs(folder_name, exist_ok=True)
        trajectory_folder = f"{folder_name}/trajectory"
        os.makedirs(trajectory_folder, exist_ok=True)
        analysis_folder = f"{folder_name}/analysis"
        os.makedirs(analysis_folder, exist_ok=True)

        traj_relaxed = path.to_traj()
        traj_rounded = path.to_traj(rounded=True) if do_rounding else None

        if save_traj:
            if traj_rounded:
                traj_rounded.save(f"{trajectory_folder}/traj_rounded.pkl")
            else:
                traj_relaxed.save(f"{trajectory_folder}/traj_relaxed.pkl")  # type: ignore

        slider_color = COLORS["aquamarine4"].diffuse()

        if traj_rounded is not None:
            make_traj_figure(
                traj_rounded,
                filename=f"{analysis_folder}/rounded_traj",
                slider_color=slider_color,
                split_on_mode_type=True,
                show_workspace=False,
            )

    return True

def render_plans(
    sim_config: PlanarPushingSimConfig,
    data_collection_config: DataCollectionConfig,
    cfg: OmegaConf,
    save_recordings: bool = False,
):
    """Renders plans according to the configs."""

    print("\nRendering plans...")

    plans = []
    for plan_dir in os.listdir(data_collection_config.plans_dir):
        if os.path.isdir(f"{data_collection_config.plans_dir}/{plan_dir}"):
            plan_path = f"{data_collection_config.plans_dir}/{plan_dir}/trajectory/traj_rounded.pkl"
            plans.append(PlanarPushingTrajectory.load(plan_path))
    
    meshcat = StartMeshcat()
    for plan in tqdm(plans):
        simulate_plan(
            traj=plan,
            sim_config=sim_config,
            data_collection_config=data_collection_config,
            cfg=cfg,
            meshcat=meshcat,
            save_recording=save_recordings,
        )
        meshcat.Delete()
        meshcat.DeleteAddedControls()


def simulate_plan(
    traj: PlanarPushingTrajectory,
    sim_config: PlanarPushingSimConfig,
    data_collection_config: DataCollectionConfig,
    cfg: OmegaConf,
    meshcat: Meshcat,
    save_recording: bool = False,
):  
    """Simulate a single plan to render the images."""

    position_source = ReplayPositionSource(
        traj=traj,
        dt = 0.025,
        delay=sim_config.delay_before_execution
    )

    ## Set up position controller
    # TODO: load with hydra instead (currently giving camera config errors)
    module_name, class_name = cfg.robot_station._target_.rsplit(".", 1)
    robot_system_class = getattr(importlib.import_module(module_name), class_name)
    position_controller: RobotSystemBase = robot_system_class(
        sim_config=sim_config, 
        meshcat=meshcat
    )

    environment = DataCollectionTableEnvironment(
        desired_position_source=position_source,
        robot_system=position_controller,
        sim_config=sim_config,
        data_collection_config=data_collection_config,
        state_estimator_meshcat=meshcat,
    )
    
    recording_name = f"recording.html" if save_recording else None
    environment.simulate(
        traj.end_time + sim_config.delay_before_execution + 0.5,
        recording_file=recording_name
    )

def convert_to_zarr(data_collection_config: DataCollectionConfig, debug: bool=False):
    """
    Converts the rendered plans to zarr format.

    This function has 2 modes (regular or reduce).
    Regular mode: Assume the rendered plans trajectory has the following structure

    rendered_plans_dir
    ├── 0
    ├──├──images
    ├──├──log.txt
    ├──├──planar_position_command.pkl
    ├── 1
    ...
    In regular mode, this function loops through all trajectories and saves the data to zarr format.

    Reduce mode: Assume the rendered plans trajectory has the following structure

    rendered_plans_dir
    ├── run_0
    ├──├── 0
    ├──├──├──images
    ├──├──├──log.txt
    ├──├──├──planar_position_command.pkl
    ├──├── 1
    ...
    ├── run_1
    ...

    In reduce mode, this function loops through all the runs. Each run contains trajectories.
    This mode is most likely used for MIT Supercloud where data generation is parallelized
    over multiple runs.
    """

    print("\nConverting data to zarr...")

    rendered_plans_dir = pathlib.Path(data_collection_config.rendered_plans_dir)
    zarr_path = data_collection_config.zarr_path

    # Collect all the data paths to compress into zarr format
    traj_dir_list = []
    if data_collection_config.convert_to_zarr_reduce:
        for run in os.listdir(rendered_plans_dir):
            run_path = rendered_plans_dir.joinpath(run)
            if not os.path.isdir(run_path):
                continue

            for plan in os.listdir(run_path):
                traj_dir = run_path.joinpath(plan)
                if not os.path.isdir(traj_dir):
                    continue
                traj_dir_list.append(traj_dir)
    else:
        for plan in os.listdir(rendered_plans_dir):
            traj_dir = rendered_plans_dir.joinpath(plan)
            if not os.path.isdir(traj_dir):
                continue
            traj_dir_list.append(traj_dir)

    concatenated_states = []
    concatenated_slider_states = []
    concatenated_actions = []
    concatenated_images = []
    concatenated_targets = []
    episode_ends = []
    current_end = 0
    freq = data_collection_config.policy_freq
    dt = 1 / freq

    for traj_dir in tqdm(traj_dir_list):
        image_dir = traj_dir.joinpath("images")
        traj_log_path = traj_dir.joinpath("combined_logs.pkl")
        log_path = traj_dir.joinpath("log.txt")

        # If too many IK fails, skip this rollout
        with open(log_path, "r") as f:
            if len(f.readlines()) != 0:
                ik_fails = int(f.readline().rsplit(" ", 1)[-1])
                if ik_fails > 5:
                    continue

        # load pickle file and timing variables
        combined_logs = pickle.load(open(traj_log_path, 'rb'))
        pusher_desired = combined_logs.pusher_desired
        slider_desired = combined_logs.slider_desired
        
        t = pusher_desired.t
        total_time = math.floor(t[-1] * freq) / freq
        
        # get start time
        start_idx = _get_start_idx(pusher_desired)   
        start_time = math.ceil(t[start_idx]*freq) / freq

        # get state, action, images
        current_time = start_time
        idx = start_idx
        state = []
        slider_state = []
        images = []
        while current_time < total_time:
            # state and action
            idx = _get_closest_index(t, current_time, idx)
            current_state = np.array([
                pusher_desired.x[idx], 
                pusher_desired.y[idx], 
                pusher_desired.theta[idx]
            ])
            current_slider_state = np.array([
                slider_desired.x[idx],
                slider_desired.y[idx],
                slider_desired.theta[idx]
            ])
            state.append(current_state)
            slider_state.append(current_slider_state)
        
            # image
            # This line can be simplified but it is clearer this way.
            # Image names are "{time in ms}" rounded to the nearest 100th
            image_name = round((current_time * 1000) / 100) * 100
            image_path = image_dir.joinpath(f"{int(image_name)}.png")
            img = Image.open(image_path).convert('RGB')
            img = np.asarray(img)
            images.append(img)
            if debug:
                from matplotlib import pyplot as plt
                print(f"\nCurrent time: {current_time}")
                print(f"Current index: {idx}")
                print(f"Image path: {image_path}")
                print(f"Current state: {current_state}")
                plt.imshow(img[6:-6, 6:-6, :])
                plt.show()

            # update current time
            current_time = round((current_time + dt) * freq) / freq

        state = np.array(state) # T x 3
        slider_state = np.array(slider_state) # T x 3
        action = np.array(state)[:,:2] # T x 2
        action = np.concatenate([action[1:, :], action[-1:, :]], axis=0) # shift action
        images = np.array(images)

        # get target
        target = np.array([
            pusher_desired.x[-1],
            pusher_desired.y[-1],
            pusher_desired.theta[-1]]
        )
        target = np.array([target for _ in range(len(state))])

        # update concatenated arrays
        concatenated_states.append(state)
        concatenated_slider_states.append(slider_state)
        concatenated_actions.append(action)
        concatenated_images.append(images)
        concatenated_targets.append(target)
        episode_ends.append(current_end + len(state))
        current_end += len(state)

    # save to zarr
    zarr_path = data_collection_config.zarr_path
    root = zarr.open_group(zarr_path, mode='w')
    data_group = root.create_group('data')
    meta_group = root.create_group('meta')

    # Chunk sizes optimized for read (not for supercloud storage, sorry admins)
    state_chunk_size = (data_collection_config.state_chunk_length, state.shape[1])
    slider_state_chunk_size = (data_collection_config.state_chunk_length, state.shape[1])
    action_chunk_size = (data_collection_config.action_chunk_length, action.shape[1])
    target_chunk_size = (data_collection_config.target_chunk_length, target.shape[1])
    image_chunk_size = (data_collection_config.image_chunk_length, *images[0].shape)
    
    # convert to numpy
    concatenated_states = np.concatenate(concatenated_states, axis=0)
    concatenated_slider_states = np.concatenate(concatenated_slider_states, axis=0)
    concatenated_actions = np.concatenate(concatenated_actions, axis=0)
    concatenated_images = np.concatenate(concatenated_images, axis=0)
    concatenated_targets = np.concatenate(concatenated_targets, axis=0)
    episode_ends = np.array(episode_ends)
    
    assert episode_ends[-1] == concatenated_states.shape[0]
    assert concatenated_states.shape[0] == concatenated_slider_states.shape[0]
    assert concatenated_states.shape[0] == concatenated_actions.shape[0]
    assert concatenated_states.shape[0] == concatenated_images.shape[0]
    assert concatenated_states.shape[0] == concatenated_targets.shape[0]

    data_group.create_dataset(
        'state', 
        data=concatenated_states, 
        chunks=state_chunk_size
    )
    data_group.create_dataset(
        'slider_state',
        data=concatenated_slider_states,
        chunks=slider_state_chunk_size
    )
    data_group.create_dataset(
        'action', 
        data=concatenated_actions, 
        chunks=action_chunk_size
    )
    data_group.create_dataset(
        'img', 
        data=concatenated_images, 
        chunks=image_chunk_size
    )
    data_group.create_dataset(
        'target', 
        data=concatenated_targets, 
        chunks=target_chunk_size
    )
    meta_group.create_dataset(
        'episode_ends', 
        data=episode_ends
    )

def _get_start_idx(pusher_desired):
    """
    Finds the index of the first "non-stationary" command.
    This is the index of the start of the trajectory.
    """

    length = len(pusher_desired.t)
    first_non_zero_idx = 0
    for i in range(length):
        if pusher_desired.x[i] != 0 or pusher_desired.y[i] != 0 or pusher_desired.theta[i] != 0:
            first_non_zero_idx = i
            break
    
    initial_state = np.array([
        pusher_desired.x[first_non_zero_idx], 
        pusher_desired.y[first_non_zero_idx], 
        pusher_desired.theta[first_non_zero_idx]
    ])
    assert not np.allclose(initial_state, np.array([0.0, 0.0, 0.0]))

    for i in range(first_non_zero_idx+1, length):
        state = np.array([pusher_desired.x[i], pusher_desired.y[i], pusher_desired.theta[i]])
        if not np.allclose(state, initial_state):
            return i
    
    return None
    
def _get_closest_index(arr, t, start_idx=None, end_idx=None):
    """Returns index of arr that is closest to t."""

    if start_idx is None:
        start_idx = 0
    if end_idx is None:
        end_idx = len(arr)
    
    min_diff = float('inf')
    min_idx = -1
    eps = 1e-4
    for i in range(start_idx, end_idx):
        diff = abs(arr[i] - t)
        if diff > min_diff:
            return min_idx
        if diff < eps:
            return i
        if diff < min_diff:
            min_diff = diff
            min_idx = i

def _get_plan_start_and_goals_to_point(
    seed: int,
    num_plans: int,
    workspace: PlanarPushingWorkspace,
    config: PlanarPlanConfig,
    point: Tuple[float, float] = (0, 0),  # Default is origin
    init_pusher_pose: Optional[PlanarPose] = None,
    limit_rotations: bool = True,  # Use this to start with
    noise_final_pose: bool = False,
) -> List[PlanarPushingStartAndGoal]:
    """Get start and goal pairs for planar pushing task"""

    # We want the plans to always be the same
    np.random.seed(seed)

    slider = config.slider_geometry

    # Hardcoded pusher start pose to be at top edge
    # of workspace
    ws = workspace.slider.new_workspace_with_buffer(new_buffer=0)
    if init_pusher_pose is not None:
        pusher_pose = init_pusher_pose
    else:
        pusher_pose = PlanarPose(ws.x_min, 0, 0)

    plans = []
    for _ in range(num_plans):
        slider_initial_pose = get_slider_pose_within_workspace(
            workspace, slider, pusher_pose, config, limit_rotations
        )

        if noise_final_pose:
            tran_tol = 0.01 # 0.01cm
            rot_tol = 1 * np.pi / 180 # 1 degrees
            slider_target_pose = PlanarPose(
                point[0] + np.random.uniform(-tran_tol, tran_tol),
                point[1] + np.random.uniform(-tran_tol, tran_tol),
                0 + np.random.uniform(-rot_tol, rot_tol),
            )
        else:
            slider_target_pose = PlanarPose(point[0], point[1], 0)

        plans.append(
            PlanarPushingStartAndGoal(
                slider_initial_pose, slider_target_pose, pusher_pose, pusher_pose
            )
        )

    return plans

def _print_data_collection_config_info(data_collection_config: DataCollectionConfig):
    """Output diagnostic info about the data collection configuration."""

    print("This data collection script is configured to perform the following steps.\n")
    step_num = 1
    if data_collection_config.generate_plans:
        print(f"{step_num}. Generate new plans in '{data_collection_config.plans_dir}' "
              f"according to the following config:")        
        print(data_collection_config.plan_config, end="\n\n")
        step_num += 1
    if data_collection_config.render_plans:
        print(f"{step_num}. Render the plans in '{data_collection_config.plans_dir}' "
              f"to '{data_collection_config.rendered_plans_dir}'\n")
        step_num += 1
    if data_collection_config.convert_to_zarr:
        print(f"{step_num}. Convert the rendered plans in '{data_collection_config.rendered_plans_dir}' "
              f"to zarr format in '{data_collection_config.zarr_path}'")
        if data_collection_config.convert_to_zarr_reduce:
            print("Converting to zarr in 'reduce' mode (i.e. performing the reduce step of map-reduce)")
            print("The 'convert_to_zarr_reduce = True' flag is usually only set for Supercloud runs.")
        print()
        step_num += 1

def _print_sim_config_info(sim_config: PlanarPushingSimConfig):
    """Output diagnostic info about the simulation configuration."""

    print(f"Initial finger pose: {sim_config.pusher_start_pose}")
    print(f"Target slider pose: {sim_config.slider_goal_pose}")
    print()

def _create_directory(dir_path):
    """Helper function for creating directories."""

    if os.path.exists(dir_path):
        user_input = input(f"{dir_path} already exists. Delete existing directory? (y/n)\n")
        if user_input.lower() != "y":
            print("Exiting")
            exit()
        shutil.rmtree(dir_path)
    else:
        os.makedirs(dir_path)

if __name__ == "__main__":
    main()