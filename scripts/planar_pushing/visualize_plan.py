from planning_through_contact.experiments.utils import get_box, get_tee
from planning_through_contact.geometry.planar.planar_pushing_trajectory import (
    PlanarPushingTrajectory,
)
from planning_through_contact.visualize.planar import (
    visualize_planar_pushing_trajectory_legacy,
)


def visualize_plan(debug: bool = False):
    plan = "trajectories/t_pusher_pushing_2.pkl"
    traj = PlanarPushingTrajectory.load(plan)

    if "box" in plan:
        slider = get_box()
    elif "t_pusher" in plan:
        slider = get_tee()
    else:
        raise NotImplementedError()

    visualize_planar_pushing_trajectory_legacy(
        traj.to_old_format(), slider.geometry, traj.pusher_radius
    )


if __name__ == "__main__":
    visualize_plan(debug=True)
