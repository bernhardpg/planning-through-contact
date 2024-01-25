import logging
from typing import List, Optional

import numpy as np
from pydrake.all import (
    ContactModel,
    RigidTransform,
    RollPitchYaw,
    RotationMatrix,
    StartMeshcat,
)

from planning_through_contact.geometry.collision_geometry.box_2d import Box2d
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.planar.planar_pushing_trajectory import (
    PlanarPushingTrajectory,
)
from planning_through_contact.simulation.controllers.cylinder_actuated_station import (
    CylinderActuatedStation,
)
from planning_through_contact.simulation.controllers.hybrid_mpc import HybridMpcConfig
from planning_through_contact.simulation.controllers.iiwa_hardware_station import (
    IiwaHardwareStation,
)
from planning_through_contact.simulation.controllers.mpc_position_source import (
    MPCPositionSource,
)
from planning_through_contact.simulation.controllers.teleop_position_source import (
    TeleopPositionSource,
)
from planning_through_contact.simulation.environments.table_environment import (
    TableEnvironment,
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
        delay_before_execution=4,
        closed_loop=True,
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
    )
    X_W_pB = RigidTransform([0.3, -0.04285714, 0.019528])
    X_W_oB = RigidTransform(
        RollPitchYaw(
            roll=0.0014019521180919092,
            pitch=-0.0017132056231440778,
            yaw=2.5206443933848894,
        ),
        [0.30213178, -0.05107934, 0.02950026],
    )

    X_oB_pB = X_W_oB.inverse() @ X_W_pB
    optitrack_config: OptitrackConfig = OptitrackConfig(
        iiwa_id=4, slider_id=10, X_optitrackBody_plantBody=X_oB_pB
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
        optitrack_config=optitrack_config,
        station_meshcat=station_meshcat,
        state_estimator_meshcat=state_estimator_meshcat,
    )
    recording_name = (
        plan.split(".")[0]
        + f"_hw_{sim_config.use_hardware}_cl{sim_config.closed_loop}"
        + ".html"
        if save_recording
        else None
    )
    # environment.export_diagram("environment_diagram.pdf")
    environment.simulate(
        traj.end_time + sim_config.delay_before_execution + 0.5,
        save_recording_as=recording_name,
    )
    # environment.simulate(
    #     sim_config.delay_before_execution+0.5,
    #     save_recording_as=recording_name,
    # )
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
    start: Optional[int] = None,
    end: Optional[int] = None,
    incl: Optional[List[int]] = None,
    station_meshcat=None,
    state_estimator_meshcat=None,
    run_rounded=True,
    run_relaxed=False,
):
    if incl is not None:
        plan_indices = incl
    else:
        plan_indices = list(range(start, end + 1))
    plans = []
    if run_rounded:
        plans = [
            f"trajectories/hw_demos_20240124130732_tee_lam_buff_04/hw_demo_{i}/trajectory/traj_rounded.pkl"
            for i in plan_indices
        ]
    if run_relaxed:
        plans += [
            f"trajectories/hw_demos_20240124130732_tee_lam_buff_04/hw_demo_{i}/trajectory/traj_relaxed.pkl"
            for i in plan_indices
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
    #     incl=[1,2,5,7,8,9,10,13,14,16],
    #     run_rounded=False,
    #     run_relaxed=True,
    #     station_meshcat=station_meshcat,
    #     state_estimator_meshcat=state_estimator_meshcat,
    # )
    run_sim(
        plan="trajectories/hw_demos_20240124130732_tee_lam_buff_04/hw_demo_9/trajectory/traj_rounded.pkl",
        # plan="trajectories/box_pushing_demos/hw_demo_C_3_rounded.pkl",
        save_recording=True,
        debug=True,
        station_meshcat=station_meshcat,
        state_estimator_meshcat=state_estimator_meshcat,
    )
    # run_sim(plan="trajectories/box_pushing_513.pkl", save_recording=True, debug=True)
