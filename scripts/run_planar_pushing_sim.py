# Import local LCM build
import sys

from pydrake.multibody.plant import ContactModel

sys.path.append("/Users/bernhardpg/software/lcm/build/python")
import lcm  # make sure we can import lcm

from planning_through_contact.simulation.planar_pushing.planar_pushing_mock_iiwa import (
    PlanarPose,
    PlanarPushingMockSimulation,
    PlanarPushingSimConfig,
)


def run_sim(debug: bool = False):
    config = PlanarPushingSimConfig(
        body="box",
        contact_model=ContactModel.kHydroelastic,
        start_pose=PlanarPose(x=0.0, y=0.5, theta=0.0),
        goal_pose=PlanarPose(x=-0.3, y=0.5, theta=0.5),
        visualize_desired=True,
    )
    sim = PlanarPushingMockSimulation(config)

    sim.reset()
    sim.run(1e-6)  # advance the sim to we can see anything

    if debug:
        sim.export_diagram("diagram.png")

    sim.run()


if __name__ == "__main__":
    run_sim(debug=False)
