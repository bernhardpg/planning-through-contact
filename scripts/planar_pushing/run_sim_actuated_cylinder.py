import numpy as np
from pydrake.multibody.plant import ContactModel

from planning_through_contact.geometry.collision_geometry.box_2d import Box2d
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.planar.planar_pushing_trajectory import (
    PlanarPushingTrajectory,
)
from planning_through_contact.geometry.rigid_body import RigidBody
from planning_through_contact.simulation.controllers.cylinder_actuated_controller import CylinderActuatedController
from planning_through_contact.simulation.controllers.teleop_position_source import TeleopPositionSource
from planning_through_contact.simulation.controllers.hybrid_mpc import HybridMpcConfig
from planning_through_contact.simulation.environments.table_environment import TableEnvironment
from planning_through_contact.simulation.planar_pushing.planar_pushing_diagram import (
    PlanarPushingSimConfig,
)
from planning_through_contact.simulation.planar_pushing.planar_pushing_sim import (
    PlanarPushingSimulation,
)
from planning_through_contact.visualize.analysis import plot_control_sols_vs_time
from scripts.planar_pushing.create_plan import get_slider_box, get_tee


def run_sim(plan: str, save_recording: bool = False, debug: bool = False):
    traj = PlanarPushingTrajectory.load(plan)

    slider = traj.config.dynamics_config.slider

    mpc_config = HybridMpcConfig(
        step_size=0.03,
        horizon=35,
        num_sliding_steps=5,
        rate_Hz=50,
        Q=np.diag([3, 3, 0.1, 0]) * 10,
        Q_N=np.diag([3, 3, 0.1, 0]) * 2000,
        R=np.diag([1, 1, 0]) * 0.5,
    )
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
        closed_loop=True,
        mpc_config=mpc_config,
        dynamics_config=traj.config.dynamics_config,
        save_plots=True,
        scene_directive_name="planar_pushing_cylinder_plant_hydroelastic.yaml"
    )
    # Choose position source
    # Option 1: Use teleop
    teleop = dict(input_limit= 1.0, step_size=0.01, start_translation=[0.0,0.0])
    position_source = TeleopPositionSource(sim_config=sim_config, teleop_config=teleop)
    # Option 2: Use open/closed loop controller based on planned trajectory
    # position_source = 
    # Set up position controller
    position_controller = CylinderActuatedController(sim_config=sim_config)
    
    environment = TableEnvironment(desired_position_source=position_source, 
                                   position_controller=position_controller,
                                   sim_config=sim_config)
    environment.setup()
    recording_name = plan.split(".")[0]+f"_cl{sim_config.closed_loop}" + ".html" if save_recording else None
    recording_name= "test_recording.html"
    environment.simulate(traj.end_time + 1, save_recording_as=recording_name)
    

if __name__ == "__main__":
    run_sim(plan="trajectories/box_pushing_demos/hw_demo_0_rounded.pkl", save_recording=True, debug=True)
    # run_sim(plan="trajectories/box_pushing_513.pkl", save_recording=True, debug=True)
