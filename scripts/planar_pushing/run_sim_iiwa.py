import numpy as np
from pydrake.all import ContactModel, StartMeshcat
import logging

from planning_through_contact.geometry.collision_geometry.box_2d import Box2d
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.planar.planar_pushing_trajectory import (
    PlanarPushingTrajectory,
)
from planning_through_contact.simulation.controllers.cylinder_actuated_station import (
    CylinderActuatedStation,
)
from planning_through_contact.simulation.controllers.iiwa_hardware_station import (
    IiwaHardwareStation,
)
from planning_through_contact.simulation.controllers.mpc_position_source import (
    MPCPositionSource,
)
from planning_through_contact.simulation.controllers.teleop_position_source import (
    TeleopPositionSource,
)
from planning_through_contact.simulation.controllers.hybrid_mpc import HybridMpcConfig
from planning_through_contact.simulation.environments.table_environment import (
    TableEnvironment,
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
    sim_config = PlanarPushingSimConfig(
        slider=slider,
        contact_model=ContactModel.kHydroelastic,
        pusher_start_pose=traj.initial_pusher_planar_pose,
        slider_start_pose=traj.initial_slider_planar_pose + disturbance,
        slider_goal_pose=traj.target_slider_planar_pose,
        visualize_desired=True,
        draw_frames=True,
        time_step=1e-3,
        use_realtime=True,
        delay_before_execution=1,
        closed_loop=False,
        mpc_config=mpc_config,
        dynamics_config=traj.config.dynamics_config,
        save_plots=True,
        scene_directive_name="planar_pushing_iiwa_plant_hydroelastic.yaml",
        use_hardware=False,
        pusher_z_offset=0.1,
    )
    # Commented out code for generating values for hybrid MPC tests
    # for t in [4, 8]:
    #     print(traj.get_slider_planar_pose(t))
    #     print(traj.get_mode(t))
    # for seg in traj.traj_segments:
    #     print(f"Segment with mode {seg.mode}, {type(seg)}, from {seg.start_time} to {seg.end_time}")

    ## Choose position source
    # Option 1: Use teleop
    # teleop = dict(input_limit= 1.0, step_size=0.01, start_translation=[0.0,0.0])
    # position_source = TeleopPositionSource(sim_config=sim_config, teleop_config=teleop, meshcat=station_meshcat)

    # Option 2: Use open/closed loop controller based on planned trajectory
    position_source = MPCPositionSource(sim_config=sim_config, traj=traj)

    ## Set up position controller
    position_controller = IiwaHardwareStation(
        sim_config=sim_config, meshcat=station_meshcat
    )

    environment = TableEnvironment(
        desired_position_source=position_source,
        robot_system=position_controller,
        sim_config=sim_config,
        station_meshcat=station_meshcat,
        state_estimator_meshcat=state_estimator_meshcat,
    )
    recording_name = (
        plan.split(".")[0] + f"_cl{sim_config.closed_loop}" + ".html"
        if save_recording
        else None
    )
    # environment.export_diagram("environment_diagram.pdf")
    environment.simulate(traj.end_time + 0.5, save_recording_as=recording_name)
    # environment.simulate(3, save_recording_as=recording_name)

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
    start: int,
    end: int,
    station_meshcat=None,
    state_estimator_meshcat=None,
    run_non_rounded=False,
):
    plans = [
        f"trajectories/box_pushing_demos/hw_demo_C_{i}_rounded.pkl"
        for i in range(start, end + 1)
    ]
    if run_non_rounded:
        plans += [
            f"trajectories/box_pushing_demos/hw_demo_C_{i}.pkl"
            for i in range(start, end + 1)
        ]
    print(f"Running {len(plans)} plans\n{plans}")
    for plan in plans:
        run_sim(
            plan,
            save_recording=True,
            debug=False,
            station_meshcat=station_meshcat,
            state_estimator_meshcat=state_estimator_meshcat,
        )
        station_meshcat.Delete()
        station_meshcat.DeleteAddedControls()
        state_estimator_meshcat.Delete()


if __name__ == "__main__":
    print(f"station meshcat")
    station_meshcat = StartMeshcat()
    print(f"state estimator meshcat")
    state_estimator_meshcat = StartMeshcat()
    # run_multiple(
    #     0,
    #     9,
    #     station_meshcat=station_meshcat,
    #     state_estimator_meshcat=state_estimator_meshcat,
    # )
    run_sim(
        # plan="trajectories/t_pusher_pushing_demos/hw_demo_C_1_rounded.pkl",
        plan="trajectories/box_pushing_demos/hw_demo_C_8_rounded.pkl",
        save_recording=True,
        debug=True,
        station_meshcat=station_meshcat,
        state_estimator_meshcat=state_estimator_meshcat,
    )
    # run_sim(plan="trajectories/box_pushing_513.pkl", save_recording=True, debug=True)
