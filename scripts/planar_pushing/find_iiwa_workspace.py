import numpy as np
import matplotlib.pyplot as plt

from pydrake.multibody.plant import ContactModel
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.planar.planar_pushing_trajectory import (
    PlanarPushingTrajectory,
)
from planning_through_contact.simulation.controllers.hybrid_mpc import HybridMpcConfig
from planning_through_contact.simulation.planar_pushing.planar_pushing_sim_config import (
    PlanarPushingSimConfig,
)

from planning_through_contact.simulation.sim_utils import LoadRobotOnly
from planning_through_contact.simulation.planar_pushing.inverse_kinematics import (
    solve_ik,
)


def run_sim(plan: str, save_recording: bool = False, debug: bool = False):
    traj = PlanarPushingTrajectory.load(plan)
    print(f"running plan:{plan}")
    print(traj.config.dynamics_config)
    slider = traj.config.dynamics_config.slider
    mpc_config = HybridMpcConfig()
    sim_config = PlanarPushingSimConfig(
        slider=slider,
        contact_model=ContactModel.kHydroelastic,
        pusher_start_pose=traj.initial_pusher_planar_pose,
        slider_start_pose=traj.initial_slider_planar_pose,
        slider_goal_pose=traj.target_slider_planar_pose,
        visualize_desired=True,
        time_step=1e-3,
        use_realtime=True,
        delay_before_execution=1,
        closed_loop=True,
        mpc_config=mpc_config,
        dynamics_config=traj.config.dynamics_config,
        save_plots=False,
        scene_directive_name="planar_pushing_iiwa_plant_hydroelastic.yaml",
        use_hardware=False,
    )
    robot_plant = LoadRobotOnly(sim_config, "iiwa_controller_plant.yaml")
    step = 0.05
    x_lim = [0, 1.2]
    y_lim = [-1, 1]
    for z in [0.03]:  # [0.03, 0.1, 0.2, 0.3]
        pass_coords = []
        fail_coords = []
        plt.figure()
        for x in np.arange(x_lim[0], x_lim[1], step):
            for y in np.arange(y_lim[0], y_lim[1], step):
                pusher_start_pose = PlanarPose(x, y, 0)

                try:
                    desired_pusher_pose = pusher_start_pose.to_pose(z)
                    start_joint_positions = solve_ik(
                        plant=robot_plant,
                        pose=desired_pusher_pose,
                        default_joint_positions=sim_config.default_joint_positions,
                    )
                    # print(f"ik succeeded for {pusher_start_pose.vector()}")
                    pass_coords.append((x, y))
                except:
                    # print(f"ik failed for {pusher_start_pose.vector()}")
                    fail_coords.append((x, y))

        # Separate the coordinates into x and y components for plotting
        pass_x, pass_y = zip(*pass_coords)
        fail_x, fail_y = zip(*fail_coords)

        # Plotting
        plt.scatter(pass_x, pass_y, color="green", label="Pass")
        plt.scatter(fail_x, fail_y, color="red", label="Fail")
        # Make axes equal length
        plt.axis("scaled")
        # Add grid and labels every 0.1
        plt.xticks(np.arange(x_lim[0], x_lim[1], 0.1))
        plt.yticks(np.arange(y_lim[0], y_lim[1], 0.1))
        # Set the figure size
        plt.gcf().set_size_inches(10, 10)
        plt.grid()
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.title(f"iiwa Workspace (z={z})")
        plt.legend()
        # Save the figure
        plt.savefig(f"iiwa_workspace_z={z}.png")


if __name__ == "__main__":
    run_sim(
        plan="trajectories/t_pusher_pushing_demos/hw_demo_C_1_rounded.pkl",
        save_recording=True,
        debug=True,
    )
