import numpy as np
import matplotlib.pyplot as plt

from pydrake.multibody.plant import ContactModel
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.planar.planar_pushing_trajectory import (
    PlanarPushingTrajectory,
)
from planning_through_contact.geometry.rigid_body import RigidBody
from planning_through_contact.simulation.controllers.hybrid_mpc import HybridMpcConfig
from planning_through_contact.simulation.planar_pushing.planar_pushing_diagram import (
    PlanarPushingSimConfig,
)
from planning_through_contact.simulation.planar_pushing.planar_pushing_sim import (
    PlanarPushingSimulation,
)
from scripts.planar_pushing.create_plan import get_slider_box, get_tee
from planning_through_contact.simulation.planar_pushing.pusher_pose_to_joint_pos import (
    solve_ik,
)


def run_sim(plan: str, save_recording: bool = False, debug: bool = False):
    traj = PlanarPushingTrajectory.load(plan)

    slider = traj.config.dynamics_config.slider

    mpc_config = HybridMpcConfig(rate_Hz=20, horizon=20, step_size=0.05)
    sim_config = PlanarPushingSimConfig(
                slider=slider,
                contact_model=ContactModel.kHydroelastic,
                pusher_start_pose=traj.initial_pusher_planar_pose,
                slider_start_pose=traj.initial_slider_planar_pose,
                slider_goal_pose=traj.target_slider_planar_pose,
                visualize_desired=True,
                time_step=1e-3,
                use_realtime=False,
                delay_before_execution=1,
                use_diff_ik=True,
                closed_loop= True,
                mpc_config=mpc_config,
                dynamics_config=traj.config.dynamics_config,
                save_plots=True,
            )
    sim = PlanarPushingSimulation(traj, sim_config)
    desired_slider_pose = sim_config.slider_start_pose.to_pose(sim.station.get_slider_min_height())
    for z in [0.05]:
        pass_coords = []
        fail_coords = []
        for x in np.arange(0, 1, 0.05):
            for y in np.arange(-1, 1, 0.05):
                pusher_start_pose = PlanarPose(x,y,0)
                
                try:
                    desired_pusher_pose = pusher_start_pose.to_pose(z)
                    start_joint_positions = solve_ik(
                        sim.diagram,
                        sim.station,
                        pose=desired_pusher_pose,
                        current_slider_pose=desired_slider_pose,
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
        plt.scatter(pass_x, pass_y, color='green', label='Pass')
        plt.scatter(fail_x, fail_y, color='red', label='Fail')
        # Make axes equal length
        plt.axis('scaled')
        # Add grid and labels every 0.1
        plt.xticks(np.arange(0, 1, 0.1))
        plt.yticks(np.arange(-1, 1, 0.1))
        # Set the figure size
        plt.gcf().set_size_inches(10, 10)
        plt.grid()
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title(f'iiwa Workspace (z={z})')
        plt.legend()
        # Save the figure
        plt.savefig(f'iiwa_workspace_z={z}.png')


if __name__ == "__main__":
    run_sim(plan="trajectories/box_pushing_513.pkl", save_recording=True, debug=True)
