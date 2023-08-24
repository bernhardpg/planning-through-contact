from planning_through_contact.geometry.collision_geometry.box_2d import Box2d
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.planar.planar_pushing_trajectory import (
    PlanarPushingTrajectory,
)
from planning_through_contact.geometry.rigid_body import RigidBody
from planning_through_contact.planning.planar.planar_plan_specs import PlanarPlanSpecs
from planning_through_contact.planning.planar.planar_pushing_planner import (
    PlanarPushingPlanner,
)
from planning_through_contact.visualize.planar import (
    visualize_planar_pushing_trajectory,
)


def create_plan(debug: bool = False):
    specs = PlanarPlanSpecs(num_knot_points_non_collision=4)

    mass = 0.1
    # TODO(bernhardpg): Make the object geometry an argument
    box_geometry = Box2d(width=0.15, height=0.15)
    body = RigidBody("box", box_geometry, mass)

    planner = PlanarPushingPlanner(
        body, specs, avoid_object=True, use_redundant_dynamic_constraints=False
    )

    # TODO(bernhardpg): Make the initial and target pose configurable
    box_initial_pose = PlanarPose(x=0.0, y=0.5, theta=0.0)
    box_target_pose = PlanarPose(x=0.5, y=0.7, theta=0.5)
    finger_initial_pose = PlanarPose(x=0.7, y=0.3, theta=0.0)
    finger_target_pose = PlanarPose(x=0.7, y=0.3, theta=0.0)

    planner.set_initial_poses(finger_initial_pose, box_initial_pose)
    planner.set_target_poses(finger_target_pose, box_target_pose)

    traj = planner.plan_trajectory(
        round_trajectory=False, print_output=debug, measure_time=debug
    )
    traj.save("box_pushing.pkl")

    if debug:
        visualize_planar_pushing_trajectory(traj.to_old_format(), box_geometry)


if __name__ == "__main__":
    create_plan(debug=True)
