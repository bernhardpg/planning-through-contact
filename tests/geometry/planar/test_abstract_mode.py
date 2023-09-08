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
from planning_through_contact.geometry.planar.trajectory_builder import (
    PlanarTrajectoryBuilder,
)
from planning_through_contact.geometry.rigid_body import RigidBody
from planning_through_contact.planning.planar.planar_plan_config import PlanarPlanConfig
from planning_through_contact.tools.gcs_tools import get_gcs_solution_path_vertices
from planning_through_contact.visualize.analysis import save_gcs_graph_diagram
from planning_through_contact.visualize.planar import (
    visualize_planar_pushing_trajectory,
)
from tests.geometry.planar.fixtures import box_geometry, gcs_options, rigid_body_box
from tests.geometry.planar.tools import assert_initial_and_final_poses


def test_add_continuity_constraints_between_non_collision_modes(
    rigid_body_box: RigidBody,
) -> None:
    config = PlanarPlanConfig(num_knot_points_non_collision=2)

    contact_location_start = PolytopeContactLocation(ContactLocation.FACE, 3)
    contact_location_end = PolytopeContactLocation(ContactLocation.FACE, 0)

    source_mode = NonCollisionMode.create_from_plan_spec(
        contact_location_start, config, rigid_body_box, "source"
    )
    target_mode = NonCollisionMode.create_from_plan_spec(
        contact_location_end, config, rigid_body_box, "target"
    )

    slider_pose = PlanarPose(0.3, 0, 0)
    source_mode.set_slider_pose(slider_pose)
    target_mode.set_slider_pose(slider_pose)

    finger_initial_pose = PlanarPose(-0.2, 0, 0)
    source_mode.set_finger_initial_pose(finger_initial_pose)
    finger_final_pose = PlanarPose(0.3, 0.5, 0)
    target_mode.set_finger_final_pose(finger_final_pose)

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

    result = gcs.SolveShortestPath(source_vertex, target_vertex)
    assert result.is_success()

    pairs = {
        "source": VertexModePair(source_vertex, source_mode),
        "target": VertexModePair(target_vertex, target_mode),
    }
    traj = PlanarTrajectoryBuilder.from_result(
        result, gcs, source_vertex, target_vertex, pairs
    ).get_trajectory(interpolate=False)

    assert_initial_and_final_poses(
        traj,
        slider_pose,
        finger_initial_pose,
        slider_pose,
        finger_final_pose,
    )

    DEBUG = False
    if DEBUG:
        save_gcs_graph_diagram(gcs, Path("test_continuity.svg"))
        save_gcs_graph_diagram(gcs, Path("test_continuity_result.svg"), result)
        visualize_planar_pushing_trajectory(traj, rigid_body_box.geometry)


@pytest.mark.parametrize(
    "rigid_body_box, gcs_options, use_eq_elimination",
    [({}, {}, False), ({}, {}, True)],
    indirect=["rigid_body_box", "gcs_options"],
    ids=["normal", "eq_elimination"],
)
def test_add_continuity_between_non_coll_and_face_contact(
    rigid_body_box: RigidBody,
    gcs_options: opt.GraphOfConvexSetsOptions,
    use_eq_elimination: bool,
) -> None:
    loc = PolytopeContactLocation(ContactLocation.FACE, 3)
    config = PlanarPlanConfig()

    slider_initial_pose = PlanarPose(0.3, 0, 0)
    slider_final_pose = PlanarPose(0.5, 0, 0.4)
    finger_final_pose = PlanarPose(-0.4, 0, 0)

    source_mode = FaceContactMode.create_from_plan_spec(loc, config, rigid_body_box)
    source_mode.set_slider_initial_pose(slider_initial_pose)
    source_mode.set_finger_pos(0.5)

    target_mode = NonCollisionMode.create_from_plan_spec(loc, config, rigid_body_box)
    target_mode.set_slider_pose(slider_final_pose)
    target_mode.set_finger_final_pose(finger_final_pose)

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
    traj = PlanarTrajectoryBuilder.from_result(
        result, gcs, source_vertex, target_vertex, pairs
    ).get_trajectory(interpolate=False)

    assert_initial_and_final_poses(
        traj,
        slider_initial_pose,
        None,
        slider_final_pose,
        finger_final_pose,
    )

    DEBUG = False
    if DEBUG:
        save_gcs_graph_diagram(gcs, Path("test_continuity.svg"))
        visualize_planar_pushing_trajectory(traj, rigid_body_box.geometry)
