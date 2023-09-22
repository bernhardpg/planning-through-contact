from planning_through_contact.geometry.collision_geometry.box_2d import Box2d
from planning_through_contact.geometry.planar.planar_pushing_trajectory import (
    PlanarPushingTrajectory,
)
from planning_through_contact.visualize.planar import (
    visualize_planar_pushing_trajectory_legacy,
)
from scripts.planar_pushing.create_plan import get_slider_box, get_tee


def visualize_plan(debug: bool = False):
    plan = "trajectories/t_pusher_pushing_2.pkl"
    traj = PlanarPushingTrajectory.load(plan)

    if "box" in plan:
        slider = get_slider_box()
        body = "box"
    elif "t_pusher" in plan:
        slider = get_tee()
        body = "t_pusher"
    else:
        raise NotImplementedError()

    visualize_planar_pushing_trajectory_legacy(
        traj.to_old_format(), slider.geometry, traj.pusher_radius
    )


if __name__ == "__main__":
    visualize_plan(debug=True)
