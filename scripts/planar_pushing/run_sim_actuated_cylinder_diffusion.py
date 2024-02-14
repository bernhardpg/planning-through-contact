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

from planning_through_contact.visualize.analysis import (
    plot_control_sols_vs_time,
    plot_cost,
    plot_velocities,
)


def run_sim(
    plan: str, # TODO: remove the need for this argument
    checkpoint: str = None,
    diffusion_policy_path: str = "/home/adam/workspace/gcs-diffusion",
    initial_pusher_planar_pose: PlanarPose = None,
    target_slider_planar_pose: PlanarPose = None,
    data_collection_dir: str = None,
    save_recording: bool = False,
    debug: bool = False,
    station_meshcat=None,
):
    logging.basicConfig(level=logging.INFO)
    logging.getLogger(
        "planning_through_contact.simulation.planar_pushing.pusher_pose_controller"
    ).setLevel(logging.DEBUG)
    traj = PlanarPushingTrajectory.load(plan)
    print(traj.config.dynamics_config)
    slider = traj.config.dynamics_config.slider

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
        fps=10.0
    )

    num_runs = 10
    max_attempt_duration = 50.0
    multi_run_config = MultiRunConfig(
        initial_slider_poses=[traj.initial_slider_planar_pose for _ in range(num_runs)],
        target_slider_poses=[traj.target_slider_planar_pose for _ in range(num_runs)],
        max_attempt_duration=max_attempt_duration
    )

    sim_config = PlanarPushingSimConfig(
        slider=slider,
        contact_model=ContactModel.kHydroelastic,
        pusher_start_pose=traj.initial_pusher_planar_pose,
        slider_start_pose=traj.initial_slider_planar_pose,
        slider_goal_pose=traj.target_slider_planar_pose,
        visualize_desired=True,
        draw_frames=True,
        time_step=1e-3,
        use_realtime=False,
        delay_before_execution=1,
        closed_loop=False,
        dynamics_config=traj.config.dynamics_config,
        save_plots=False,
        scene_directive_name="planar_pushing_cylinder_plant_hydroelastic.yaml",
        pusher_z_offset=0.03,
        camera_config=camera_config,
        collect_data=False,
        data_dir='diffusion_policy_logs',
        multi_run_config=multi_run_config
    )
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
    environment.simulate(end_time, recording_file=recording_name)
    # environment.simulate(10, save_recording_as=recording_name)
    environment.save_data("diffusion_policy_logs")


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


if __name__ == "__main__":
    print(f"station meshcat")
    station_meshcat = StartMeshcat()
    # plan path is used to extract sim_config
    # the trajectory in plan path is not used
    plan = "data_collection_trajectories/run_0/traj_0/trajectory/traj_rounded.pkl"
    run_sim(
        plan=plan,
        # checkpoint='/home/adam/workspace/gcs-diffusion/data/outputs/push_tee_v1_sc/checkpoints/epoch_148.ckpt',
        # checkpoint='/home/adam/workspace/gcs-diffusion/data/outputs/push_tee_v2/checkpoints/working_better.ckpt',
        checkpoint='/home/adam/workspace/gcs-diffusion/data/outputs/push_tee_v2/checkpoints/epoch=0695-val_loss=0.035931.ckpt',
        data_collection_dir=None,
        save_recording=True,
        debug=False,
        station_meshcat=station_meshcat,
    )