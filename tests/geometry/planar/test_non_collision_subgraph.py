from pathlib import Path

import numpy as np
import pydrake.geometry.optimization as opt

from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    ContactLocation,
    PolytopeContactLocation,
)
from planning_through_contact.geometry.planar.face_contact import FaceContactMode
from planning_through_contact.geometry.planar.non_collision import NonCollisionMode
from planning_through_contact.geometry.planar.non_collision_subgraph import (
    NonCollisionSubGraph,
    VertexModePair,
    gcs_add_edge_with_continuity,
)
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.planar.trajectory_builder import (
    PlanarTrajectoryBuilder,
)
from planning_through_contact.geometry.rigid_body import RigidBody
from planning_through_contact.planning.planar.planar_plan_specs import PlanarPlanSpecs
from planning_through_contact.tools.gcs_tools import get_gcs_solution_path
from planning_through_contact.visualize.analysis import save_gcs_graph_diagram
from planning_through_contact.visualize.planar import (
    visualize_planar_pushing_trajectory,
)
from tests.geometry.planar.fixtures import (
    box_geometry,
    initial_and_final_non_collision_mode,
    rigid_body_box,
)


def test_single_non_collision_subgraph_DEPRECATED(rigid_body_box: RigidBody):
    plan_specs = PlanarPlanSpecs()
    gcs = opt.GraphOfConvexSets()

    contact_location_start = PolytopeContactLocation(ContactLocation.FACE, 0)
    contact_location_end = PolytopeContactLocation(ContactLocation.FACE, 1)

    contact_modes = [
        FaceContactMode.create_from_plan_spec(loc, plan_specs, rigid_body_box)
        for loc in (contact_location_start, contact_location_end)
    ]

    contact_vertices = [
        gcs.AddVertex(mode.get_convex_set(), mode.name) for mode in contact_modes
    ]

    non_collision_modes = [
        NonCollisionMode.create_from_plan_spec(loc, plan_specs, rigid_body_box)
        for loc in rigid_body_box.geometry.contact_locations
    ]

    subgraph = NonCollisionSubGraph.from_modes(
        non_collision_modes, gcs, contact_modes[0], contact_modes[1]
    )
    breakpoint()

    assert len(gcs.Vertices()) == len(contact_modes) + len(non_collision_modes)

    # Edges are bi-directional
    expected_num_edges = len(non_collision_modes) * 2
    assert len(gcs.Edges()) == expected_num_edges

    # Adds another 4 edges to the graph
    subgraph.connect_to_contact_vertex(gcs, contact_vertices[0], 0)
    subgraph.connect_to_contact_vertex(gcs, contact_vertices[1], 1)
    assert len(gcs.Edges()) == expected_num_edges + 4


def test_single_non_collision_subgraph(rigid_body_box: RigidBody):
    plan_specs = PlanarPlanSpecs()
    gcs = opt.GraphOfConvexSets()

    subgraph = NonCollisionSubGraph.create_with_gcs(
        gcs, rigid_body_box, plan_specs, "Subgraph_TEST"
    )

    assert len(gcs.Vertices()) == len(subgraph.non_collision_modes)

    # Edges are bi-directional
    assert len(gcs.Edges()) == len(subgraph.non_collision_modes) * 2

    contact_location_start = PolytopeContactLocation(ContactLocation.FACE, 3)
    contact_location_end = PolytopeContactLocation(ContactLocation.FACE, 0)

    source_mode = NonCollisionMode.create_from_plan_spec(
        contact_location_start, plan_specs, rigid_body_box, "source"
    )
    target_mode = NonCollisionMode.create_from_plan_spec(
        contact_location_end, plan_specs, rigid_body_box, "target"
    )

    slider_pose = PlanarPose(0.3, 0, 0)
    source_mode.set_slider_pose(slider_pose)
    target_mode.set_slider_pose(slider_pose)

    finger_initial_pose = PlanarPose(-0.2, 0, 0)
    source_mode.set_finger_initial_pos(finger_initial_pose.pos())
    finger_final_pose = PlanarPose(-0, 0.4, 0)
    target_mode.set_finger_final_pos(finger_final_pose.pos())

    source_vertex = gcs.AddVertex(source_mode.get_convex_set(), source_mode.name)
    target_vertex = gcs.AddVertex(target_mode.get_convex_set(), target_mode.name)

    subgraph.connect_with_continuity_constraints(
        3, VertexModePair(source_vertex, source_mode)
    )
    subgraph.connect_with_continuity_constraints(
        0, VertexModePair(target_vertex, target_mode)
    )

    result = gcs.SolveShortestPath(source_vertex, target_vertex)
    assert result.is_success()

    # TODO(bernhardpg): use new Drake function:
    # edges = gcs.GetSolutionPath(source, target, result)

    pairs = subgraph.get_all_vertex_mode_pairs()
    pairs["source"] = VertexModePair(source_vertex, source_mode)
    pairs["target"] = VertexModePair(target_vertex, target_mode)
    vertex_path = get_gcs_solution_path(gcs, result, source_vertex, target_vertex)
    pairs_on_path = [pairs[v.name()] for v in vertex_path]
    path = [
        pair.mode.get_variable_solutions_for_vertex(pair.vertex, result)
        for pair in pairs_on_path
    ]

    DEBUG = True
    if DEBUG:
        save_gcs_graph_diagram(gcs, Path("subgraph.svg"))
        save_gcs_graph_diagram(gcs, Path("subgraph_result.svg"), result)
        traj = PlanarTrajectoryBuilder(path).get_trajectory(interpolate=False)
        visualize_planar_pushing_trajectory(traj, rigid_body_box.geometry)

    # contact_location_start = PolytopeContactLocation(ContactLocation.FACE, 0)
    # contact_location_end = PolytopeContactLocation(ContactLocation.FACE, 1)

    # contact_modes = [
    #     FaceContactMode.create_from_plan_spec(loc, plan_specs, rigid_body_box)
    #     for loc in (contact_location_start, contact_location_end)
    # ]
    # contact_vertices = [
    #     gcs.AddVertex(mode.get_convex_set(), mode.name) for mode in contact_modes
    # ]


# if __name__ == "__main__":
#     # test_single_non_collision_subgraph(rigid_body_box(box_geometry()))
