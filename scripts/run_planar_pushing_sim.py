# Import local LCM build
import sys

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
)
from planning_through_contact.visualize.planar import (
    visualize_planar_pushing_trajectory,
)


def run_sim(debug: bool = False):
    sim = PlanarPushingMockSimulation()

    # NOTE: These are currently hardcoded to match the pre-computed plan
    box_initial_pose = PlanarPose(x=0.0, y=0.4, theta=0.0)
    sim.set_box_planar_pose(box_initial_pose)

    sim.reset()
    sim.run(1e-6)  # advance the sim to we can see anything

    if debug:
        sim.export_diagram("diagram.png")

    sim.run()


if __name__ == "__main__":
    run_sim(debug=False)
