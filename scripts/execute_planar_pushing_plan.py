from pydrake.multibody.plant import ContactModel

from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.planar.planar_pushing_trajectory import (
    PlanarPushingTrajectory,
)
from planning_through_contact.simulation.planar_pushing.planar_pushing_diagram import (
    PlanarPushingSimConfig,
)
from planning_through_contact.simulation.planar_pushing.planar_pushing_sim import (
    PlanarPushingSimulation,
)


def run_sim(plan: str, debug: bool = False):
    traj_name = "trajectories/box_pushing_2.pkl"
    # TODO: Retrieve start pose and goal pose from trajectory
    traj = PlanarPushingTrajectory.load(traj_name)

    config = PlanarPushingSimConfig(
        body="box",
        contact_model=ContactModel.kHydroelastic,
        start_pose=PlanarPose(x=0.0, y=0.5, theta=0.0),
        goal_pose=PlanarPose(x=-0.3, y=0.5, theta=0.5),
        visualize_desired=True,
    )

    sim = PlanarPushingSimulation(traj, config)

    # sim.reset()
    # sim.run(1e-6)  # advance the sim to we can see anything

    if debug:
        sim.export_diagram("diagram.png")

    # sim.run()


if __name__ == "__main__":
    run_sim(plan="trajectories/box_pushing_1.pkl", debug=True)
