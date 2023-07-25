import pydrake.geometry.optimization as opt

from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    ContactLocation,
    PolytopeContactLocation,
)
from planning_through_contact.geometry.planar.face_contact import FaceContactMode
from planning_through_contact.geometry.planar.non_collision import NonCollisionMode
from planning_through_contact.geometry.planar.non_collision_subgraph import (
    NonCollisionSubGraph,
)
from planning_through_contact.geometry.rigid_body import RigidBody
from planning_through_contact.planning.planar.planar_plan_specs import PlanarPlanSpecs
from tests.geometry.planar.fixtures import box_geometry, rigid_body_box


def test_single_non_collision_subgraph(rigid_body_box: RigidBody):
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

    subgraph = NonCollisionSubGraph.from_modes(non_collision_modes, gcs, 0, 1)

    assert len(gcs.Vertices()) == len(contact_modes) + len(non_collision_modes)

    # Edges are bi-directional
    expected_num_edges = len(non_collision_modes) * 2
    assert len(gcs.Edges()) == expected_num_edges

    # Adds another 4 edges to the graph
    subgraph.add_connection_to_full_graph(gcs, contact_vertices[0], 0)
    subgraph.add_connection_to_full_graph(gcs, contact_vertices[1], 1)
    assert len(gcs.Edges()) == expected_num_edges + 4


#     breakpoint()
#
#
# if __name__ == "__main__":
#     test_single_non_collision_subgraph(rigid_body_box(box_geometry()))
