import numpy as np
import numpy.typing as npt
from pydrake.solvers import Solve

from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    PolytopeContactLocation,
)
from planning_through_contact.geometry.planar.non_collision import NonCollisionMode
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.rigid_body import RigidBody
from planning_through_contact.planning.planar.planar_plan_specs import PlanarPlanSpecs


def find_first_matching_location(
    finger_pose: PlanarPose,
    slider_pose: PlanarPose,
    slider: RigidBody,
) -> PolytopeContactLocation:
    # we always add all non-collision modes, even when we don't add all contact modes
    # (think of maneuvering around the object etc)
    locations = slider.geometry.contact_locations
    matching_locs = [
        loc
        for loc in locations
        if check_finger_pose_in_contact_location(finger_pose, loc, slider, slider_pose)
    ]
    if len(matching_locs) == 0:
        raise ValueError(
            "No valid configurations found for specified initial or target poses"
        )
    return matching_locs[0]


def check_finger_pose_in_contact_location(
    finger_pose: PlanarPose,
    loc: PolytopeContactLocation,
    body: RigidBody,
    body_pose: PlanarPose,
) -> bool:
    specs = PlanarPlanSpecs()
    mode = NonCollisionMode.create_from_plan_spec(loc, specs, body, one_knot_point=True)

    mode.set_finger_initial_pose(finger_pose)
    mode.set_slider_pose(body_pose)

    result = Solve(mode.prog)
    return result.is_success()
