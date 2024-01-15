from pathlib import Path

import numpy as np
import pydrake.geometry.optimization as opt
import pytest
from pydrake.solvers import CommonSolverOption, SolverOptions

from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    ContactLocation,
    PolytopeContactLocation,
)
from planning_through_contact.geometry.planar.face_contact import FaceContactMode
from planning_through_contact.geometry.planar.non_collision import NonCollisionMode
from planning_through_contact.geometry.planar.non_collision_subgraph import (
    VertexModePair,
    gcs_add_edge_with_continuity,
)
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.planar.planar_pushing_path import (
    PlanarPushingPath,
)
from planning_through_contact.geometry.planar.planar_pushing_trajectory import (
    PlanarPushingTrajectory,
)
from planning_through_contact.geometry.planar.trajectory_builder import (
    PlanarTrajectoryBuilder,
)
from planning_through_contact.geometry.rigid_body import RigidBody
from planning_through_contact.planning.planar.planar_plan_config import (
    PlanarPlanConfig,
    PlanarPushingStartAndGoal,
    SliderPusherSystemConfig,
)
from planning_through_contact.tools.gcs_tools import get_gcs_solution_path_vertices
from planning_through_contact.visualize.analysis import save_gcs_graph_diagram
from planning_through_contact.visualize.planar_pushing import (
    visualize_planar_pushing_trajectory,
    visualize_planar_pushing_trajectory_legacy,
)
from tests.geometry.planar.fixtures import (
    box_geometry,
    dynamics_config,
    gcs_options,
    plan_config,
    rigid_body_box,
)
from tests.geometry.planar.tools import (
    assert_initial_and_final_poses,
    assert_initial_and_final_poses_LEGACY,
)

DEBUG = False


def test_add_continuity_constraints_between_non_collision_modes(
    dynamics_config: SliderPusherSystemConfig,
) -> None:
    config = PlanarPlanConfig(
        num_knot_points_non_collision=2, dynamics_config=dynamics_config
    )

    contact_location_start = PolytopeContactLocation(ContactLocation.FACE, 3)
    contact_location_end = PolytopeContactLocation(ContactLocation.FACE, 0)

    source_mode = NonCollisionMode.create_from_plan_spec(
        contact_location_start, config, "source"
    )
    target_mode = NonCollisionMode.create_from_plan_spec(
        contact_location_end, config, "target"
    )

    slider_pose = PlanarPose(0.0, 0, 0)
    source_mode.set_slider_pose(slider_pose)
    target_mode.set_slider_pose(slider_pose)

    finger_initial_pose = PlanarPose(-0.2, 0, 0)
    source_mode.set_finger_initial_pose(finger_initial_pose)
    finger_target_pose = PlanarPose(0.0, 0.2, 0)
    target_mode.set_finger_final_pose(finger_target_pose)

    gcs = opt.GraphOfConvexSets()
    source_vertex = gcs.AddVertex(source_mode.get_convex_set(), source_mode.name)
    target_vertex = gcs.AddVertex(target_mode.get_convex_set(), target_mode.name)

    gcs_add_edge_with_continuity(
        gcs,
        VertexModePair(source_vertex, source_mode),
        VertexModePair(target_vertex, target_mode),
    )

    gcs_add_edge_with_continuity(
        gcs,
        VertexModePair(target_vertex, target_mode),
        VertexModePair(source_vertex, source_mode),
    )

    save_gcs_graph_diagram(gcs, Path("test_continuity.svg"))
    result = gcs.SolveShortestPath(source_vertex, target_vertex)
    assert result.is_success()

    pairs = {
        "source": VertexModePair(source_vertex, source_mode),
        "target": VertexModePair(target_vertex, target_mode),
    }

    path = PlanarPushingPath.from_result(
        gcs, result, source_vertex, target_vertex, pairs
    )
    traj = path.to_traj()

    # NOTE: This only works because the slider pose is (0,0,0)
    # as the finger poses are in the B frame
    assert_initial_and_final_poses(
        traj,
        slider_pose,
        finger_initial_pose,
        slider_pose,
        finger_target_pose,
    )

    if DEBUG:
        start_and_goal = PlanarPushingStartAndGoal(
            slider_pose, slider_pose, finger_initial_pose, finger_target_pose
        )
        traj.config.start_and_goal = start_and_goal  # needed for viz
        save_gcs_graph_diagram(gcs, Path("test_continuity.svg"))
        save_gcs_graph_diagram(gcs, Path("test_continuity_result.svg"), result)
        visualize_planar_pushing_trajectory(
            traj, visualize_knot_points=True, save=True, filename="debug_file"
        )


def test_add_velocity_constraints_between_non_collision_modes(
    dynamics_config: SliderPusherSystemConfig,
) -> None:
    # We need at least 3 knot points for this to work
    config = PlanarPlanConfig(
        num_knot_points_non_collision=3, dynamics_config=dynamics_config
    )

    contact_location_start = PolytopeContactLocation(ContactLocation.FACE, 3)
    contact_location_end = PolytopeContactLocation(ContactLocation.FACE, 0)

    source_mode = NonCollisionMode.create_from_plan_spec(
        contact_location_start, config, "source"
    )
    target_mode = NonCollisionMode.create_from_plan_spec(
        contact_location_end, config, "target"
    )

    slider_pose = PlanarPose(0.0, 0, 0)
    source_mode.set_slider_pose(slider_pose)
    target_mode.set_slider_pose(slider_pose)

    finger_initial_pose = PlanarPose(-0.2, 0, 0)
    source_mode.set_finger_initial_pose(finger_initial_pose)
    finger_target_pose = PlanarPose(0.0, 0.2, 0)
    target_mode.set_finger_final_pose(finger_target_pose)

    gcs = opt.GraphOfConvexSets()
    source_vertex = gcs.AddVertex(source_mode.get_convex_set(), source_mode.name)
    target_vertex = gcs.AddVertex(target_mode.get_convex_set(), target_mode.name)

    gcs_add_edge_with_continuity(
        gcs,
        VertexModePair(source_vertex, source_mode),
        VertexModePair(target_vertex, target_mode),
        continuity_on_pusher_velocities=True,
    )

    gcs_add_edge_with_continuity(
        gcs,
        VertexModePair(target_vertex, target_mode),
        VertexModePair(source_vertex, source_mode),
        continuity_on_pusher_velocities=True,
    )

    save_gcs_graph_diagram(gcs, Path("test_continuity.svg"))
    result = gcs.SolveShortestPath(source_vertex, target_vertex)
    assert result.is_success()

    pairs = {
        "source": VertexModePair(source_vertex, source_mode),
        "target": VertexModePair(target_vertex, target_mode),
    }

    path = PlanarPushingPath.from_result(
        gcs, result, source_vertex, target_vertex, pairs
    )
    traj = path.to_traj()

    # NOTE: This only works because the slider pose is (0,0,0)
    # as the finger poses are in the B frame
    assert_initial_and_final_poses(
        traj,
        slider_pose,
        finger_initial_pose,
        slider_pose,
        finger_target_pose,
    )

    if DEBUG:
        start_and_goal = PlanarPushingStartAndGoal(
            slider_pose, slider_pose, finger_initial_pose, finger_target_pose
        )
        traj.config.start_and_goal = start_and_goal  # needed for viz
        save_gcs_graph_diagram(gcs, Path("test_continuity.svg"))
        save_gcs_graph_diagram(gcs, Path("test_continuity_result.svg"), result)
        visualize_planar_pushing_trajectory(
            traj, visualize_knot_points=True, save=True, filename="debug_file"
        )


def test_add_continuity_between_non_coll_and_face_contact(
    plan_config: PlanarPlanConfig,
    gcs_options: opt.GraphOfConvexSetsOptions,
) -> None:
    plan_config.num_knot_points_non_collision = 4
    loc = PolytopeContactLocation(ContactLocation.FACE, 3)

    slider_initial_pose = PlanarPose(0.3, 0, 0)
    slider_target_pose = PlanarPose(0.5, 0, 0.4)
    finger_target_pose = PlanarPose(-0.4, 0, 0)

    source_mode = FaceContactMode.create_from_plan_spec(loc, plan_config)
    source_mode.set_slider_initial_pose(slider_initial_pose)
    # source_mode.set_finger_pos(0.5) # We cannot set finger position, because then it may be infeasible.

    target_mode = NonCollisionMode.create_from_plan_spec(loc, plan_config)
    target_mode.set_slider_pose(slider_target_pose)
    target_mode.set_finger_final_pose(finger_target_pose)

    gcs = opt.GraphOfConvexSets()
    source_vertex = gcs.AddVertex(source_mode.get_convex_set(), source_mode.name)
    target_vertex = gcs.AddVertex(target_mode.get_convex_set(), target_mode.name)

    source_mode.add_cost_to_vertex(source_vertex)
    target_mode.add_cost_to_vertex(target_vertex)

    source = VertexModePair(source_vertex, source_mode)
    target = VertexModePair(target_vertex, target_mode)
    gcs_add_edge_with_continuity(gcs, source, target)

    result = gcs.SolveShortestPath(source_vertex, target_vertex, gcs_options)
    assert result.is_success()

    pairs = {source.vertex.name(): source, target.vertex.name(): target}

    path = PlanarPushingPath.from_result(
        gcs, result, source_vertex, target_vertex, pairs
    )
    traj = path.to_traj()

    assert_initial_and_final_poses(
        traj,
        slider_initial_pose,
        None,
        slider_target_pose,
        None,
    )

    # Assert p_BP pose like this because it is in the B frame
    assert np.allclose(traj.get_value(traj.end_time, "p_BP"), finger_target_pose.pos())

    if DEBUG:
        start_and_goal = PlanarPushingStartAndGoal(
            slider_initial_pose, slider_target_pose, None, finger_target_pose
        )
        traj.config.start_and_goal = start_and_goal  # needed for viz
        save_gcs_graph_diagram(gcs, Path("test_continuity.svg"))
        save_gcs_graph_diagram(gcs, Path("test_continuity_result.svg"), result)
        visualize_planar_pushing_trajectory(
            traj, visualize_knot_points=True, save=True, filename="debug_file"
        )
