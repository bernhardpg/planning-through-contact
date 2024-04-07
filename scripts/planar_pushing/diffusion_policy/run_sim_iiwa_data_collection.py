import logging
from typing import List, Optional
import argparse
import os
from tqdm import tqdm

import numpy as np
from pydrake.all import (
    ContactModel,
    RigidTransform,
    RollPitchYaw,
    RotationMatrix,
    StartMeshcat,
)
from pydrake.systems.sensors import (
    CameraConfig
)
from pydrake.common.schema import (
    Transform
)

from planning_through_contact.geometry.collision_geometry.box_2d import Box2d
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.planar.planar_pushing_trajectory import (
    PlanarPushingTrajectory,
)
from planning_through_contact.simulation.controllers.hybrid_mpc import HybridMpcConfig
from planning_through_contact.simulation.controllers.iiwa_hardware_station import (
    IiwaHardwareStation,
)
from planning_through_contact.simulation.controllers.mpc_position_source import (
    MPCPositionSource,
)

from planning_through_contact.simulation.environments.data_collection_table_environment import (
    DataCollectionTableEnvironment,
)
from planning_through_contact.simulation.planar_pushing.planar_pushing_sim_config import (
    PlanarPushingSimConfig,
)
from planning_through_contact.simulation.sensors.optitrack_config import OptitrackConfig
from planning_through_contact.visualize.analysis import (
    plot_control_sols_vs_time,
    plot_cost,
    plot_velocities,
)
from planning_through_contact.simulation.controllers.replay_position_source import (
    ReplayPositionSource,
)

def run_sim(
    plan: str,
    save_recording: bool = False,
    debug: bool=False,
    station_meshcat=None,
    state_estimator_meshcat=None,
    data_collection_dir=None,
):
    logging.basicConfig(level=logging.INFO)
    logging.getLogger(
        "planning_through_contact.simulation.planar_pushing.pusher_pose_controller"
    ).setLevel(logging.DEBUG)
    logging.getLogger(
        "planning_through_contact.simulation.controllers.hybrid_mpc"
    ).setLevel(logging.DEBUG)
    logging.getLogger(
        "planning_through_contact.simulation.planar_pushing.iiwa_planner"
    ).setLevel(logging.DEBUG)
    logging.getLogger(
        "planning_through_contact.simulation.environments.table_environment"
    ).setLevel(logging.DEBUG)

    traj = PlanarPushingTrajectory.load(plan)
    print(f"running plan:{plan}")
    # traj.config.dynamics_config.integration_constant = 0.1
    print(traj.config.dynamics_config)

    slider = traj.config.dynamics_config.slider
    disturbance = PlanarPose(x=0.0, y=0, theta=0)
    # note that MPC config is not actualyl used
    # it is only used to create the MPCPositionSource
    mpc_config = HybridMpcConfig(
        step_size=0.03,
        horizon=35,
        num_sliding_steps=1,
        rate_Hz=50,
        Q=np.diag([3, 3, 0.1, 0]) * 100,
        Q_N=np.diag([3, 3, 1, 0]) * 2000,
        R=np.diag([1, 1, 0.1]) * 0.5,
        u_max_magnitude=[4, 4, 2],
        lam_max=0.8,
        lam_min=0.2,
    )
    # camera set up

    print(f"Initial finger pose: {traj.initial_pusher_planar_pose}")
    print(f"Target slider pose: {traj.target_slider_planar_pose}")

    zoom = 1.0
    position = np.array([0.5 + traj.target_slider_planar_pose.x, 0, 0.5]) / zoom
    center_of_view = np.array([traj.target_slider_planar_pose.x, 0.0, 0.0])
    angle = 0.9*np.arctan((position[0]-center_of_view[0])/(position[2]-center_of_view[2]))
    orientation = RollPitchYaw(0, np.pi - angle, np.pi)
    camera_config = CameraConfig(
        name="overhead_camera",
        X_PB=Transform(
            RigidTransform(orientation, position)
        ),
        width=128,
        height=128,
        show_rgb=False,
    )
    sim_config = PlanarPushingSimConfig(
        slider=slider,
        contact_model=ContactModel.kHydroelastic,
        pusher_start_pose=traj.initial_pusher_planar_pose,
        slider_start_pose=traj.initial_slider_planar_pose + disturbance,
        slider_goal_pose=traj.target_slider_planar_pose,
        visualize_desired=True,
        draw_frames=True,
        time_step=1e-3,
        use_realtime=False,
        delay_before_execution=5,
        closed_loop=False,
        mpc_config=mpc_config,
        dynamics_config=traj.config.dynamics_config,
        save_plots=True,
        scene_directive_name="planar_pushing_iiwa_plant_hydroelastic.yaml",
        use_hardware=False,
        pusher_z_offset=0.03,
        default_joint_positions=[
            0.0776,
            1.0562,
            0.3326,
            -1.3048,
            2.7515,
            -0.8441,
            0.5127,
        ],
        # Adam's additions
        camera_configs=[camera_config],
        collect_data=True,
        data_dir = data_collection_dir
    )

    position_source = ReplayPositionSource(
        traj=traj,
        dt = 0.025,
        delay=sim_config.delay_before_execution
    )

    ## Set up position controller
    position_controller = IiwaHardwareStation(
        sim_config=sim_config, meshcat=station_meshcat
    )

    environment = DataCollectionTableEnvironment(
        desired_position_source=position_source,
        robot_system=position_controller,
        sim_config=sim_config,
        state_estimator_meshcat=state_estimator_meshcat,
    )
    recording_name = (
        plan.split(".")[0]
        + f"_hw_{sim_config.use_hardware}_cl{sim_config.closed_loop}"
        + ".html"
        if save_recording
        else None
    )
    recording_name = "iiwa_bad_training_example.html"

    environment.export_diagram("environment_diagram.pdf")
    environment.simulate(
        traj.end_time + sim_config.delay_before_execution + 0.5,
        recording_file=recording_name,
    )

def run_multiple(
    plans: list,
    save_dir: str,
    station_meshcat=None, 
    state_estimator_meshcat=None
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
            state_estimator_meshcat=state_estimator_meshcat,
        )
        station_meshcat.Delete()
        station_meshcat.DeleteAddedControls()
        state_estimator_meshcat.Delete()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--trajectory_dir", type=str, default=None)
    args = parser.parse_args()

    if args.trajectory_dir is None:
        print(f"station meshcat")
        station_meshcat = StartMeshcat()
        print(f"state estimator meshcat")
        state_estimator_meshcat = StartMeshcat()
        run_sim(
            plan="trajectories/data_collection_trajectories_tee_v1/run_0/traj_0/trajectory/traj_rounded.pkl",
            save_recording=True,
            debug=False,
            station_meshcat=station_meshcat,
            state_estimator_meshcat=state_estimator_meshcat,
            data_collection_dir='temp'
        )
    else:
        traj_dir = args.trajectory_dir
        list_dir = os.listdir(traj_dir)
        plans = []
        for name in list_dir:
            if os.path.isdir(os.path.join(traj_dir, name)):
                plan = os.path.join(traj_dir, name, "trajectory", "traj_rounded.pkl")
                plans.append(plan)
        # note that plans is not stored in numerical order
        # i.e. index i is not necessarily the i-th plan
                
        print(f"station meshcat")
        station_meshcat = StartMeshcat()
        print(f"state estimator meshcat")
        state_estimator_meshcat = StartMeshcat()

        run_multiple(
            plans=plans,
            save_dir=args.save_dir,
            station_meshcat=station_meshcat,
            state_estimator_meshcat=state_estimator_meshcat,
        )