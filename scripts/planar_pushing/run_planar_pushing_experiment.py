from copy import copy
import logging
import os
from datetime import datetime

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd, instantiate
from omegaconf import OmegaConf, open_dict

from pydrake.all import (
    StartMeshcat,
)

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
from planning_through_contact.simulation.environments.table_environment import (
    TableEnvironment,
)
from planning_through_contact.simulation.planar_pushing.planar_pushing_sim_config import (
    PlanarPushingSimConfig,
)
from planning_through_contact.simulation.sensors.optitrack_config import OptitrackConfig
from planning_through_contact.simulation.sensors.realsense_camera_config import (
    RealsenseCameraConfig,
)

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../config", config_name="basic")
def main(cfg: OmegaConf) -> None:
    now = datetime.now()
    now.strftime("%Y-%m-%d_%H-%M-%S")
    # Add log dir to config
    hydra_config = HydraConfig.get()
    full_log_dir = hydra_config.runtime.output_dir

    with open_dict(cfg):
        cfg.log_dir = os.path.relpath(full_log_dir, get_original_cwd() + "/outputs")

    logger.info(OmegaConf.to_yaml(cfg))

    plan_folder = (
        f"trajectories/{cfg.trajectory_set_name}/" + f"hw_demo_{cfg.trajectory_index}"
    )
    traj_name = f"traj_{'rounded' if cfg.use_rounded else 'relaxed'}"
    traj_file = f"{plan_folder}/trajectory/" + f"{traj_name}.pkl"

    # Copy plan folder to log dir
    os.system(f"cp -r {plan_folder} {full_log_dir}")

    # Set up config data structures
    try:
        traj = PlanarPushingTrajectory.load(traj_file)
    except FileNotFoundError:
        logger.error(f"Trajectory file {traj_file} not found")
        return
    mpc_config: HybridMpcConfig = instantiate(cfg.mpc_config)
    optitrack_config: OptitrackConfig = instantiate(cfg.optitrack_config)

    sim_config: PlanarPushingSimConfig = PlanarPushingSimConfig.from_traj(
        trajectory=traj, mpc_config=mpc_config, **cfg.sim_config
    )

    station_meshcat = StartMeshcat()
    state_estimator_meshcat = StartMeshcat()
    logger.info(f"station meshcat url {station_meshcat.web_url()}")
    logger.info(f"state estimator meshcat url {state_estimator_meshcat.web_url()}")

    if sim_config.use_hardware:
        reset_experiment(
            sim_config, traj, station_meshcat, state_estimator_meshcat, optitrack_config
        )

    # Initialize position source
    position_source = MPCPositionSource(sim_config=sim_config, traj=traj)

    # Initialize robot system
    position_controller = IiwaHardwareStation(
        sim_config=sim_config, meshcat=station_meshcat
    )

    # Initialize environment
    environment = TableEnvironment(
        desired_position_source=position_source,
        robot_system=position_controller,
        sim_config=sim_config,
        optitrack_config=optitrack_config,
        station_meshcat=station_meshcat,
        state_estimator_meshcat=state_estimator_meshcat,
    )

    try:
        if sim_config.use_hardware and cfg.realsense_config.should_record:
            from planning_through_contact.simulation.sensors.realsense import (
                RealsenseCamera,
            )

            # Initialize cameras
            camera_config: RealsenseCameraConfig = instantiate(
                cfg.realsense_config.realsense_camera_config
            )
            camera_config.output_dir = full_log_dir
            camera1 = RealsenseCamera(
                name=cfg.realsense_config.camera1_name,
                serial_number=cfg.realsense_config.camera1_serial_number,
                config=camera_config,
            )
            camera2 = RealsenseCamera(
                name=cfg.realsense_config.camera2_name,
                serial_number=cfg.realsense_config.camera2_serial_number,
                config=camera_config,
            )
            camera3 = RealsenseCamera(
                name=cfg.realsense_config.camera3_name,
                serial_number=cfg.realsense_config.camera3_serial_number,
                config=camera_config,
            )
            camera1.start_recording()
            camera2.start_recording()
            camera3.start_recording()

        recording_name = os.path.join(
            full_log_dir,
            traj_name
            + f"_hw_{sim_config.use_hardware}_cl{sim_config.closed_loop}"
            + ".html"
            if cfg.save_experiment_data
            else None,
        )
        timeout = (
            cfg.override_duration
            if "override_duration" in cfg
            else traj.end_time + sim_config.delay_before_execution
        )

        # Run simulation
        environment.simulate(
            timeout=timeout,
            recording_file=recording_name,
            save_dir=full_log_dir,
        )

        if sim_config.use_hardware and cfg.realsense_config.should_record:
            camera1.stop_recording()
            camera2.stop_recording()
            camera3.stop_recording()
    except KeyboardInterrupt:
        environment.save_logs(recording_name, full_log_dir)
        if sim_config.use_hardware and cfg.realsense_config.should_record:
            camera1.stop_recording()
            camera2.stop_recording()
            camera3.stop_recording()


def reset_experiment(
    sim_config: PlanarPushingSimConfig,
    traj,
    station_meshcat,
    state_estimator_meshcat,
    optitrack_config,
):
    reset_sim_config = copy(sim_config)
    reset_sim_config.delay_before_execution = 600
    # Initialize position source
    position_source = MPCPositionSource(sim_config=reset_sim_config, traj=traj)

    # Initialize robot system
    position_controller = IiwaHardwareStation(
        sim_config=reset_sim_config, meshcat=station_meshcat
    )

    # Initialize environment
    environment = TableEnvironment(
        desired_position_source=position_source,
        robot_system=position_controller,
        sim_config=reset_sim_config,
        optitrack_config=optitrack_config,
        station_meshcat=station_meshcat,
        state_estimator_meshcat=state_estimator_meshcat,
    )

    environment.simulate(
        timeout=reset_sim_config.delay_before_execution,
        for_reset=True,
    )

    station_meshcat.Delete()
    state_estimator_meshcat.Delete()


if __name__ == "__main__":
    main()
