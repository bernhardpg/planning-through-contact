from pathlib import Path

import numpy as np
from geometry.planar.trajectory_builder import PlanarTrajectoryBuilder
from planning.planar_pushing_planner import PlanarPlanSpecs, PlanarPushingPlanner
from simulation.planar_pushing.planar_pushing_iiwa import (
    PlanarPose,
    PlanarPushingSimulation,
)
from visualize.planar import visualize_planar_pushing_trajectory_legacy


def planar_pushing_station():
    sim = PlanarPushingSimulation()
    box = sim.get_box()

    box_initial_pose = PlanarPose(x=0.0, y=0.0, theta=0.0)
    box_target_pose = PlanarPose(x=0.5, y=0.5, theta=0.0)
    # finger_initial_pose = PlanarPose(x=0.7, y=0.3, theta=0.0)
    # finger_target_pose = PlanarPose(x=0.7, y=0.3, theta=0.0)

    specs = PlanarPlanSpecs()

    planner = PlanarPushingPlanner(box, specs)

    planner.set_slider_initial_pose(box_initial_pose)
    planner.set_slider_target_pose(box_target_pose)
    planner.save_graph_diagram(Path("graph.svg"))
    traj = planner.make_trajectory(
        interpolate=False, print_path=True, measure_time=True, print_output=False
    )

    visualize_planar_pushing_trajectory_legacy(traj, box.geometry)
    breakpoint()

    # sim.set_box_planar_pose(box_initial_pose)

    # sim.set_pusher_planar_pose(finger_initial_pose)

    finger_pose = sim.get_pusher_planar_pose()
    sim.run()


if __name__ == "__main__":
    # manipulation_station()
    # simple_iiwa_and_brick()
    planar_pushing_station()
    # teleop()
