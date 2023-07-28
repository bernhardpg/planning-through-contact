from pathlib import Path

import numpy as np
import pydrake.geometry.optimization as opt
from pydrake.solvers import LinearCost, QuadraticCost

from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    ContactLocation,
    PolytopeContactLocation,
)
from planning_through_contact.geometry.planar.face_contact import FaceContactMode
from planning_through_contact.geometry.planar.non_collision import NonCollisionMode
from planning_through_contact.geometry.planar.non_collision_subgraph import (
    NonCollisionSubGraph,
    VertexModePair,
)
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.planar.trajectory_builder import (
    PlanarTrajectoryBuilder,
)
from planning_through_contact.geometry.rigid_body import RigidBody
from planning_through_contact.planning.planar.planar_plan_specs import PlanarPlanSpecs
from planning_through_contact.planning.planar.planar_pushing_planner import (
    find_first_matching_location,
)
from planning_through_contact.visualize.analysis import save_gcs_graph_diagram
from planning_through_contact.visualize.planar import (
    visualize_planar_pushing_trajectory,
)
from tests.geometry.planar.fixtures import box_geometry, gcs_options, rigid_body_box
from tests.geometry.planar.tools import assert_initial_and_final_poses


def test_non_collision_subgraph(
    rigid_body_box: RigidBody,
):
    plan_specs = PlanarPlanSpecs()
    gcs = opt.GraphOfConvexSets()

    subgraph = NonCollisionSubGraph.create_with_gcs(
        gcs, rigid_body_box, plan_specs, "Subgraph_TEST"
    )

    assert len(gcs.Vertices()) == len(subgraph.non_collision_modes)

    # Edges are bi-directional, one edge between each mode for box
    assert len(gcs.Edges()) == len(subgraph.non_collision_modes) * 2

    num_continuity_variables = 6
    for edge in gcs.Edges():
        assert len(edge.GetConstraints()) == num_continuity_variables

    for vertex in gcs.Vertices():
        # Squared eucl distance
        costs = vertex.GetCosts()
        assert len(costs) == 1

        # p_BF for each knot point should be in the cost
        cost = costs[0]
        NUM_DIMS = 2
        assert (
            len(cost.variables()) == plan_specs.num_knot_points_non_collision * NUM_DIMS
        )

        # Squared eucl distance
        assert isinstance(cost.evaluator(), QuadraticCost)


def test_non_collision_subgraph_planning(
    rigid_body_box: RigidBody,
):
    plan_specs = PlanarPlanSpecs()
    gcs = opt.GraphOfConvexSets()

    subgraph = NonCollisionSubGraph.create_with_gcs(
        gcs, rigid_body_box, plan_specs, "Subgraph_TEST"
    )

    contact_location_start = PolytopeContactLocation(ContactLocation.FACE, 3)
    contact_location_end = PolytopeContactLocation(ContactLocation.FACE, 0)

    source_mode = NonCollisionMode.create_from_plan_spec(
        contact_location_start,
        plan_specs,
        rigid_body_box,
        "source",
        one_knot_point=True,
    )
    target_mode = NonCollisionMode.create_from_plan_spec(
        contact_location_end, plan_specs, rigid_body_box, "target", one_knot_point=True
    )

    slider_pose = PlanarPose(0.3, 0, 0)
    source_mode.set_slider_pose(slider_pose)
    target_mode.set_slider_pose(slider_pose)

    finger_initial_pose = PlanarPose(-0.2, 0, 0)
    source_mode.set_finger_initial_pos(finger_initial_pose.pos())
    finger_final_pose = PlanarPose(0.3, 0.5, 0)
    target_mode.set_finger_final_pos(finger_final_pose.pos())

    source_vertex = gcs.AddVertex(source_mode.get_convex_set(), source_mode.name)
    target_vertex = gcs.AddVertex(target_mode.get_convex_set(), target_mode.name)

    subgraph.connect_with_continuity_constraints(
        3, VertexModePair(source_vertex, source_mode)
    )
    subgraph.connect_with_continuity_constraints(
        0, VertexModePair(target_vertex, target_mode)
    )

    # We should have added 4 more edges
    assert len(gcs.Edges()) == len(subgraph.non_collision_modes) * 2 + 4

    result = gcs.SolveShortestPath(source_vertex, target_vertex)
    assert result.is_success()

    # TODO(bernhardpg): use new Drake function:
    # edges = gcs.GetSolutionPath(source, target, result)

    pairs = subgraph.get_all_vertex_mode_pairs()
    pairs["source"] = VertexModePair(source_vertex, source_mode)
    pairs["target"] = VertexModePair(target_vertex, target_mode)
    traj = PlanarTrajectoryBuilder.from_result(
        result, gcs, source_vertex, target_vertex, pairs
    ).get_trajectory(interpolate=True)

    assert_initial_and_final_poses(
        traj, slider_pose, finger_initial_pose, slider_pose, finger_final_pose
    )

    # Make sure we are not leaving the object
    assert np.all(np.abs(traj.p_c_W) <= 1.0)

    DEBUG = False
    if DEBUG:
        save_gcs_graph_diagram(gcs, Path("subgraph.svg"))
        save_gcs_graph_diagram(gcs, Path("subgraph_result.svg"), result)
        visualize_planar_pushing_trajectory(traj, rigid_body_box.geometry)


def test_subgraph_with_contact_modes(
    rigid_body_box: RigidBody, gcs_options: opt.GraphOfConvexSetsOptions
):
    plan_specs = PlanarPlanSpecs()
    gcs = opt.GraphOfConvexSets()

    subgraph = NonCollisionSubGraph.create_with_gcs(
        gcs, rigid_body_box, plan_specs, "Subgraph_TEST"
    )

    contact_location_start = PolytopeContactLocation(ContactLocation.FACE, 3)
    contact_location_end = PolytopeContactLocation(ContactLocation.FACE, 2)

    source_mode = FaceContactMode.create_from_plan_spec(
        contact_location_start, plan_specs, rigid_body_box
    )
    target_mode = FaceContactMode.create_from_plan_spec(
        contact_location_end, plan_specs, rigid_body_box
    )

    slider_initial_pose = PlanarPose(0.3, 0, 0)
    source_mode.set_slider_initial_pose(slider_initial_pose)
    source_vertex = gcs.AddVertex(source_mode.get_convex_set(), "source")
    source_mode.add_cost_to_vertex(source_vertex)

    source = VertexModePair(source_vertex, source_mode)

    slider_final_pose = PlanarPose(0.5, 0.3, 0.4)
    target_mode.set_slider_final_pose(slider_final_pose)
    target_vertex = gcs.AddVertex(target_mode.get_convex_set(), "target")
    target_mode.add_cost_to_vertex(target_vertex)
    target = VertexModePair(target_vertex, target_mode)

    subgraph.connect_with_continuity_constraints(
        contact_location_start.idx, source, outgoing=False
    )
    subgraph.connect_with_continuity_constraints(
        contact_location_end.idx, target, incoming=False
    )

    result = gcs.SolveShortestPath(source_vertex, target_vertex, gcs_options)
    assert result.is_success()

    pairs = subgraph.get_all_vertex_mode_pairs()
    pairs["source"] = source
    pairs["target"] = target

    traj = PlanarTrajectoryBuilder.from_result(
        result, gcs, source_vertex, target_vertex, pairs
    ).get_trajectory(interpolate=False)

    assert_initial_and_final_poses(
        traj, slider_initial_pose, None, slider_final_pose, None
    )

    # Make sure we are not leaving the object
    assert np.all(np.abs(traj.p_c_W) <= 1.0)

    DEBUG = False
    if DEBUG:
        save_gcs_graph_diagram(gcs, Path("subgraph_w_contact.svg"))
        visualize_planar_pushing_trajectory(traj, rigid_body_box.geometry)


def test_subgraph_with_object_avoidance(
    rigid_body_box: RigidBody,
) -> None:
    plan_specs = PlanarPlanSpecs(num_knot_points_non_collision=3)
    gcs = opt.GraphOfConvexSets()

    subgraph = NonCollisionSubGraph.create_with_gcs(
        gcs, rigid_body_box, plan_specs, "Subgraph_TEST", avoid_object=True
    )

    slider_pose = PlanarPose(0.3, 0, 0)
    finger_initial_pose = PlanarPose(-0.2, 0, 0)
    finger_final_pose = PlanarPose(0.3, 0.5, 0)

    contact_location_start = find_first_matching_location(
        finger_initial_pose.pos(), slider_pose, rigid_body_box
    )
    contact_location_end = find_first_matching_location(
        finger_final_pose.pos(), slider_pose, rigid_body_box
    )

    source_mode = NonCollisionMode.create_from_plan_spec(
        contact_location_start,
        plan_specs,
        rigid_body_box,
        "source",
        one_knot_point=True,
    )
    target_mode = NonCollisionMode.create_from_plan_spec(
        contact_location_end, plan_specs, rigid_body_box, "target", one_knot_point=True
    )

    source_mode.set_slider_pose(slider_pose)
    target_mode.set_slider_pose(slider_pose)

    source_mode.set_finger_initial_pos(finger_initial_pose.pos())
    target_mode.set_finger_final_pos(finger_final_pose.pos())

    source_vertex = gcs.AddVertex(source_mode.get_convex_set(), source_mode.name)
    target_vertex = gcs.AddVertex(target_mode.get_convex_set(), target_mode.name)

    subgraph.connect_with_continuity_constraints(
        3, VertexModePair(source_vertex, source_mode), outgoing=False
    )
    subgraph.connect_with_continuity_constraints(
        0, VertexModePair(target_vertex, target_mode), incoming=False
    )

    # Check costs are correctly added to GCS instance
    for v in gcs.Vertices():
        if v.name() in ("source", "target"):
            continue

        costs = v.GetCosts()
        assert len(costs) == 2

        # eucl distance cost
        assert isinstance(costs[0].evaluator(), QuadraticCost)

        # maximize distance cost
        assert isinstance(costs[1].evaluator(), LinearCost)

    result = gcs.SolveShortestPath(source_vertex, target_vertex)
    assert result.is_success()

    pairs = subgraph.get_all_vertex_mode_pairs()
    pairs["source"] = VertexModePair(source_vertex, source_mode)
    pairs["target"] = VertexModePair(target_vertex, target_mode)
    traj = PlanarTrajectoryBuilder.from_result(
        result, gcs, source_vertex, target_vertex, pairs
    ).get_trajectory(interpolate=False)

    assert_initial_and_final_poses(
        traj, slider_pose, finger_initial_pose, slider_pose, finger_final_pose
    )

    # Make sure we are not leaving the object
    assert np.all(np.abs(traj.p_c_W) <= 3.0)

    DEBUG = False
    if DEBUG:
        save_gcs_graph_diagram(gcs, Path("subgraph.svg"))
        save_gcs_graph_diagram(gcs, Path("subgraph_result.svg"), result)
        visualize_planar_pushing_trajectory(traj, rigid_body_box.geometry)
