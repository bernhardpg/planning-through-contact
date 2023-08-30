from pydrake.multibody.plant import ContactModel

from planning_through_contact.geometry.collision_geometry.box_2d import Box2d
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.planar.planar_pushing_trajectory import (
    PlanarPushingTrajectory,
)
from planning_through_contact.geometry.rigid_body import RigidBody
from planning_through_contact.simulation.planar_pushing.planar_pushing_diagram import (
    PlanarPushingSimConfig,
)
from planning_through_contact.simulation.planar_pushing.planar_pushing_sim import (
    PlanarPushingSimulation,
)


# TODO(bernhardpg): Generalize this
def get_slider_box() -> RigidBody:
    mass = 0.1
    box_geometry = Box2d(width=0.15, height=0.15)
    slider = RigidBody("box", box_geometry, mass)
    return slider


def run_sim(plan: str, save_recording: bool = False, debug: bool = False):
    traj_name = "trajectories/box_pushing_1.pkl"
    traj = PlanarPushingTrajectory.load(traj_name)

    slider = get_slider_box()

    config = PlanarPushingSimConfig(
        body="box",
        contact_model=ContactModel.kHydroelastic,
        start_pose=traj.initial_planar_pose,
        goal_pose=traj.target_planar_pose,
        visualize_desired=True,
        time_step=1e-3,
        use_realtime=True,
        delay_before_execution=2.0,
    )

    sim = PlanarPushingSimulation(traj, slider, config)

    sim.reset()
    recording_name = traj_name.split(".")[0] + ".html" if save_recording else None
    sim.run(traj.end_time, save_recording_as=recording_name)

    if debug:
        sim.export_diagram("diagram.png")


if __name__ == "__main__":
    run_sim(plan="trajectories/box_pushing_1.pkl", debug=True)
