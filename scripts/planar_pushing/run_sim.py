from pydrake.multibody.plant import ContactModel

from planning_through_contact.geometry.collision_geometry.box_2d import Box2d
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


def run_sim(plan: str, save_recording: bool = False, debug: bool = False):
    traj = PlanarPushingTrajectory.load(plan)

    if "box" in plan:
        slider = get_slider_box()
        body = "box"
    elif "t_pusher" in plan:
        slider = get_tee()
        body = "t_pusher"
    else:
        raise NotImplementedError()

    mpc_config = HybridMpcConfig(rate_Hz=50, horizon=20, step_size=0.05)
    config = PlanarPushingSimConfig(
        body=body,
        contact_model=ContactModel.kHydroelastic,
        pusher_start_pose=traj.initial_pusher_planar_pose,
        slider_start_pose=traj.initial_slider_planar_pose,
        slider_goal_pose=traj.target_slider_planar_pose,
        visualize_desired=True,
        time_step=1e-3,
        use_realtime=True,
        delay_before_execution=2.0,
        use_diff_ik=True,
        closed_loop=True,
        mpc_config=mpc_config,
    )

    sim = PlanarPushingSimulation(traj, slider, config)
    # if debug:
    #     sim.export_diagram("simulation_diagram.pdf")

    sim.reset()
    recording_name = plan.split(".")[0] + ".html" if save_recording else None
    sim.run(traj.end_time + 5, save_recording_as=recording_name)


if __name__ == "__main__":
    run_sim(plan="trajectories/t_pusher_pushing_6.pkl", save_recording=True, debug=True)
