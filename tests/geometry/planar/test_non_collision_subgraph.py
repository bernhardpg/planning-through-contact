from pathlib import Path
from typing import Tuple

import numpy as np
import pydrake.geometry.optimization as opt
from pydrake.solvers import QuadraticCost

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


# TODO delete
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


def test_single_non_collision_subgraph(
    rigid_body_box: RigidBody,
    initial_and_final_non_collision_mode: Tuple[NonCollisionMode, NonCollisionMode],
):
    plan_specs = PlanarPlanSpecs()
    gcs = opt.GraphOfConvexSets()

    subgraph = NonCollisionSubGraph.create_with_gcs(
        gcs, rigid_body_box, plan_specs, "Subgraph_TEST"
    )

    #####
    # TODO(bernhardpg): Break these into their own test

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

    #####

    source_mode, target_mode = initial_and_final_non_collision_mode
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
    ).get_trajectory(interpolate=False)

    p_c_W_initial = source_mode.slider_pose.pos() + source_mode.p_BF_initial
    assert np.allclose(traj.p_c_W[:, 0:1], p_c_W_initial)

    p_c_W_final = target_mode.slider_pose.pos() + target_mode.p_BF_final
    assert np.allclose(traj.p_c_W[:, -1:], p_c_W_final)

    breakpoint()

    DEBUG = True
    if DEBUG:
        save_gcs_graph_diagram(gcs, Path("subgraph.svg"))
        save_gcs_graph_diagram(gcs, Path("subgraph_result.svg"), result)
        visualize_planar_pushing_trajectory(traj, rigid_body_box.geometry)


if __name__ == "__main__":
    test_single_non_collision_subgraph(
        rigid_body_box(box_geometry()),
        initial_and_final_non_collision_mode(rigid_body_box(box_geometry())),
    )
