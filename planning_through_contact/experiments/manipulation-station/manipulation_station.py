from pathlib import Path

import numpy as np

from planning_through_contact.geometry.planar.trajectory_builder import (
    PlanarTrajectoryBuilder,
)
from planning_through_contact.planning.planar.planar_pushing_planner import (
    PlanarPlanSpecs,
    PlanarPushingPlanner,
)
from planning_through_contact.simulation.planar_pushing.planar_pushing_iiwa import (
    PlanarPose,
    PlanarPushingSimulation,
)
from planning_through_contact.visualize.planar import (
    visualize_planar_pushing_trajectory,
)


def planar_pushing_station():
    sim = PlanarPushingSimulation()
    box = sim.get_box()

    box_initial_pose = PlanarPose(x=0.4, y=0.3, theta=0.0)
    box_target_pose = PlanarPose(x=0.4, y=0.4, theta=0.0)
    finger_initial_pose = PlanarPose(x=0.4, y=0.0, theta=0.0)
    finger_target_pose = finger_initial_pose
    # finger_target_pose = PlanarPose(x=0.4, y=0.3, theta=0.0)

    specs = PlanarPlanSpecs()
    planner = PlanarPushingPlanner(box, specs, [box.geometry.contact_locations[0]])

    planner.set_initial_poses(finger_initial_pose.pos(), box_initial_pose)
    planner.set_target_poses(finger_target_pose.pos(), box_target_pose)

    traj = planner.plan_trajectory(
        interpolate=False, print_path=True, measure_time=True, print_output=False
    )

    DEBUG = True
    if DEBUG:
        planner.save_graph_diagram(Path("graph.svg"))
        visualize_planar_pushing_trajectory(traj, box.geometry)

    sim.set_box_planar_pose(box_initial_pose)
    sim.set_pusher_planar_pose(finger_initial_pose)

    finger_pose = sim.get_pusher_planar_pose()
    sim.run()
    while True:
        ...


if __name__ == "__main__":
    # manipulation_station()
    # simple_iiwa_and_brick()
    planar_pushing_station()
    # teleop()
