import numpy as np
import argparse
import os
from tqdm import tqdm

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
from planning_through_contact.simulation.controllers.mpc_position_source import (
    MPCPositionSource,
)
from planning_through_contact.simulation.controllers.hybrid_mpc import HybridMpcConfig

from planning_through_contact.simulation.environments.data_collection_table_environment import (
    DataCollectionTableEnvironment,
)
from planning_through_contact.simulation.planar_pushing.planar_pushing_sim_config import (
    PlanarPushingSimConfig,
)

from planning_through_contact.visualize.analysis import (
    plot_control_sols_vs_time,
    plot_cost,
    plot_velocities,
)


def run_sim(
    plan: str,
    data_collection_dir: str = None,
    save_recording: bool = False,
    debug: bool = False,
    station_meshcat=None,
    state_estimator_meshcat=None,
):
    logging.basicConfig(level=logging.INFO)
    logging.getLogger(
        "planning_through_contact.simulation.planar_pushing.pusher_pose_controller"
    ).setLevel(logging.DEBUG)
    logging.getLogger(
        "planning_through_contact.simulation.controllers.hybrid_mpc"
    ).setLevel(logging.DEBUG)
    traj = PlanarPushingTrajectory.load(plan)
    print(f"running plan:{plan}")
    print(traj.config.dynamics_config)
    slider = traj.config.dynamics_config.slider
    mpc_config = HybridMpcConfig(
        step_size=0.03,
        horizon=35,
        num_sliding_steps=1,
        rate_Hz=50,
        Q=np.diag([3, 3, 0.01, 0]) * 100,
        Q_N=np.diag([3, 3, 0.01, 0]) * 2000,
        R=np.diag([1, 1, 0]) * 0.5,
        u_max_magnitude=[0.3, 0.3, 0.1],
    )
    # disturbance = PlanarPose(x=0.01, y=0, theta=-15* np.pi/180)
    disturbance = PlanarPose(x=0.0, y=0, theta=0)

    # camera set up
    camera_config = CameraConfig(
        name="overhead_camera",
        X_PB=Transform(
            RigidTransform(
                RotationMatrix.MakeXRotation(np.pi),
                np.array([0.5, 0.0, 1.0])
            )
        ),
        width=96,
        height=96,
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
        delay_before_execution=1,
        closed_loop=False,
        mpc_config=mpc_config,
        dynamics_config=traj.config.dynamics_config,
        save_plots=False,
        scene_directive_name="planar_pushing_cylinder_plant_hydroelastic.yaml",
        pusher_z_offset=0.03,
        camera_config=camera_config,
        collect_data=True,
        data_dir = data_collection_dir
    )
    # Using MPCPositionSource in open loop to output the pusher and slider
    # states directly to the state estimator for data collection
    position_source = MPCPositionSource(sim_config=sim_config, traj=traj)

    ## Set up position controller
    position_controller = CylinderActuatedStation(
        sim_config=sim_config, meshcat=station_meshcat
    )

    environment = DataCollectionTableEnvironment(
        desired_position_source=position_source,
        robot_system=position_controller,
        sim_config=sim_config,
        station_meshcat=station_meshcat,
        state_estimator_meshcat=state_estimator_meshcat,
        optitrack_config=None,
    )
    recording_name = (
        plan.split(".")[0] + f"_actuated_cylinder_cl{sim_config.closed_loop}" + ".html"
        if save_recording
        else None
    )
    environment.export_diagram("environment_diagram.pdf")
    environment.simulate(traj.end_time + 0.5, recording_file=recording_name)
    # environment.simulate(10, save_recording_as=recording_name)

    if debug and isinstance(position_source, MPCPositionSource):
        for (
            contact_loc,
            mpc,
        ) in position_source._pusher_pose_controller.mpc_controllers.items():
            if len(mpc.control_log) > 0:
                plot_cost(mpc.cost_log, suffix=f"_{contact_loc}")
                plot_control_sols_vs_time(mpc.control_log, suffix=f"_{contact_loc}")
                # plot_velocities(
                #     mpc.desired_velocity_log,
                #     mpc.commanded_velocity_log,
                #     suffix=f"_{contact_loc}",
                # )


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
            # plan="data_collection_trajectories_tee/run_0/traj_0/trajectory/traj_rounded.pkl",
            plan='diffusion_policy_logs/0/combined_planar_pushing_logs.pkl',
            data_collection_dir=args.save_dir,
            save_recording=False,
            debug=False,
            station_meshcat=station_meshcat,
            state_estimator_meshcat=state_estimator_meshcat,
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