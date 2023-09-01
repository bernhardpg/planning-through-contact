from planning_through_contact.geometry.collision_geometry.box_2d import Box2d
from planning_through_contact.geometry.planar.planar_pushing_trajectory import (
    PlanarPushingTrajectory,
)
from planning_through_contact.visualize.planar import (
    visualize_planar_pushing_trajectory,
)


def visualize_plan(debug: bool = False):
    box_geometry = Box2d(width=0.15, height=0.15)
    traj = PlanarPushingTrajectory.load("trajectories/box_pushing_4.pkl")
    visualize_planar_pushing_trajectory(
        traj.to_old_format(), box_geometry, traj.pusher_radius
    )


if __name__ == "__main__":
    visualize_plan(debug=True)
