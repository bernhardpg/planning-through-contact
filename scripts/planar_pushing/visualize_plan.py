from planning_through_contact.geometry.collision_geometry.box_2d import Box2d
from planning_through_contact.geometry.planar.planar_pushing_trajectory import (
    PlanarPushingTrajectory,
)
from planning_through_contact.visualize.planar import (
    visualize_planar_pushing_trajectory,
)
from scripts.planar_pushing.create_plan import get_slider_box, get_tee


def visualize_plan(debug: bool = False):
    plan = "trajectories/sugar_box_pushing_4.pkl"
    traj = PlanarPushingTrajectory.load(plan)
    slider = traj.config.dynamics_config.slider

    visualize_planar_pushing_trajectory(
        traj.to_old_format(),
        slider.geometry,
        traj.pusher_radius,
        visualize_robot_base=True,
    )


if __name__ == "__main__":
    visualize_plan(debug=True)
