# Import local LCM build
import sys

from pydrake.multibody.plant import ContactModel

from planning_through_contact.geometry.planar.planar_pushing_trajectory import (
    PlanarPushingTrajectory,
)

sys.path.append("/Users/bernhardpg/software/lcm/build/python")
import lcm  # make sure we can import lcm

from planning_through_contact.simulation.hardware.planar_pushing_mock_iiwa import (
    PlanarPose,
    PlanarPushingHardwareMock,
    PlanarPushingSimConfig,
)


def run_hardware_mock(debug: bool = False):
    traj_name = "trajectories/box_pushing_2.pkl"
    traj = PlanarPushingTrajectory.load(traj_name)

    config = PlanarPushingSimConfig(
        body="box",
        contact_model=ContactModel.kHydroelastic,
        start_pose=traj.initial_planar_pose,
        goal_pose=traj.target_planar_pose,
        visualize_desired=True,
        time_step=1e-3,
    )
    sim = PlanarPushingHardwareMock(config)

    sim.reset()
    sim.run(1e-6)  # advance the sim to we can see anything

    if debug:
        sim.export_diagram("diagram.png")

    sim.run()


if __name__ == "__main__":
    run_hardware_mock(debug=False)
