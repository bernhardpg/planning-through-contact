# Import local LCM build
import sys

from pydrake.multibody.plant import ContactModel

sys.path.append("/Users/bernhardpg/software/lcm/build/python")
import lcm
import numpy as np

from planning_through_contact.geometry.planar.trajectory_builder import (
    OldPlanarPushingTrajectory,
    PlanarTrajectoryBuilder,
)
from planning_through_contact.planning.planar.planar_pushing_planner import (
    PlanarPlanSpecs,
    PlanarPushingPlanner,
)
from planning_through_contact.simulation.planar_pushing.planar_pushing_mock_iiwa import (
    PlanarPose,
    PlanarPushingMockSimulation,
    PlanarPushingSimConfig,
)
from planning_through_contact.visualize.planar import (
    visualize_planar_pushing_trajectory,
)


def run_sim(debug: bool = False):
    config = PlanarPushingSimConfig(
        body="t_pusher",
        contact_model=ContactModel.kHydroelastic,
        start_pose=PlanarPose(x=0.0, y=0.5, theta=0.0),
        goal_pose=PlanarPose(x=0.3, y=0.5, theta=0.5),
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
