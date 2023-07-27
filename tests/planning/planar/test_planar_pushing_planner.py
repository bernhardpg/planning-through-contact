from pathlib import Path

import numpy as np
import pydrake.geometry.optimization as opt
import pytest
from pydrake.solvers import CommonSolverOption, LinearCost, SolverOptions
from pydrake.symbolic import Variables

from planning_through_contact.geometry.collision_geometry.box_2d import Box2d
from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    ContactLocation,
    PolytopeContactLocation,
)
from planning_through_contact.geometry.planar.face_contact import FaceContactMode
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
from planning_through_contact.planning.planar.planar_pushing_planner import (
    PlanarPushingPlanner,
)
from planning_through_contact.visualize.analysis import save_gcs_graph_diagram
from planning_through_contact.visualize.planar import (
    visualize_planar_pushing_trajectory,
)
from tests.geometry.planar.fixtures import box_geometry, rigid_body_box


@pytest.fixture
def planar_pushing_planner(rigid_body_box: RigidBody) -> PlanarPushingPlanner:
    specs = PlanarPlanSpecs()
    return PlanarPushingPlanner(rigid_body_box, specs)


def test_planar_pushing_planner_construction(
    planar_pushing_planner: PlanarPushingPlanner,
) -> None:
    # One contact mode per face
    assert len(planar_pushing_planner.contact_modes) == 4

    # One contact mode per face
    assert len(planar_pushing_planner.contact_vertices) == 4

    # One subgraph between each contact mode:
    # 4 choose 2 = 6
    assert len(planar_pushing_planner.subgraphs) == 6

    for v, m in zip(
        planar_pushing_planner.contact_vertices, planar_pushing_planner.contact_modes
    ):
        costs = v.GetCosts()

        # angular velocity and linear velocity
        assert len(costs) == 2

        lin_vel, ang_vel = costs

        target_lin_vars = Variables(v.x()[m._get_cost_terms()[0][0]])
        assert target_lin_vars.EqualTo(Variables(lin_vel.variables()))

        target_ang_vars = Variables(v.x()[m._get_cost_terms()[0][1]])
        assert target_ang_vars.EqualTo(Variables(ang_vel.variables()))

        # Costs should be linear in SDP relaxation
        assert isinstance(lin_vel.evaluator(), LinearCost)
        assert isinstance(ang_vel.evaluator(), LinearCost)

    assert planar_pushing_planner.source_subgraph is not None
    assert planar_pushing_planner.target_subgraph is not None

    DEBUG = False
    if DEBUG:
        save_gcs_graph_diagram(
            planar_pushing_planner.gcs, Path("planar_pushing_graph.svg")
        )


def test_planar_pushing_planner_set_initial_and_final(
    planar_pushing_planner: PlanarPushingPlanner,
) -> None:
    finger_initial_pose = PlanarPose(x=-0.3, y=0, theta=0.0)

    box_initial_pose = PlanarPose(x=0.0, y=0.0, theta=0.0)
    box_target_pose = PlanarPose(x=0.5, y=0.5, theta=0.0)

    planar_pushing_planner.set_initial_poses(
        finger_initial_pose.pos(),
        box_initial_pose,
        PolytopeContactLocation(ContactLocation.FACE, 3),
    )

    planar_pushing_planner.set_target_poses(
        finger_initial_pose.pos(),
        box_target_pose,
        PolytopeContactLocation(ContactLocation.FACE, 3),
    )

    DEBUG = True
    if DEBUG:
        save_gcs_graph_diagram(
            planar_pushing_planner.gcs, Path("planar_pushing_graph.svg")
        )

    num_vertices_per_subgraph = 4
    num_contact_modes = 4
    num_subgraphs = 8  # 4 choose 2 + entry and exit
    expected_num_vertices = (
        num_contact_modes
        + num_vertices_per_subgraph * num_subgraphs
        + 2  # source and target vertices
    )
    assert len(planar_pushing_planner.gcs.Vertices()) == expected_num_vertices

    num_edges_per_subgraph = 8
    num_edges_per_contact_mode = 8  # 2 to all three other modes + entry and exit
    expected_num_edges = (
        num_edges_per_subgraph * num_subgraphs
        + num_edges_per_contact_mode * num_contact_modes
        + 2  # source and target
    )
    assert len(planar_pushing_planner.gcs.Edges()) == expected_num_edges


def test_subgraph_with_contact_modes(
    rigid_body_box: RigidBody,
):
    plan_specs = PlanarPlanSpecs()
    gcs = opt.GraphOfConvexSets()

    subgraph = NonCollisionSubGraph.create_with_gcs(
        gcs, rigid_body_box, plan_specs, "Subgraph_TEST"
    )

    contact_location_start = PolytopeContactLocation(ContactLocation.FACE, 3)
    contact_location_end = PolytopeContactLocation(ContactLocation.FACE, 1)

    source_mode = FaceContactMode.create_from_plan_spec(
        contact_location_start, plan_specs, rigid_body_box
    )
    target_mode = FaceContactMode.create_from_plan_spec(
        contact_location_end, plan_specs, rigid_body_box
    )

    slider_initial_pose = PlanarPose(0.3, 0, 0)
    source_mode.set_slider_initial_pose(slider_initial_pose)
    source_vertex = gcs.AddVertex(source_mode.get_convex_set(), "source")
    source = VertexModePair(source_vertex, source_mode)

    slider_final_pose = PlanarPose(0.5, 0, 0)
    target_mode.set_slider_final_pose(slider_final_pose)
    target_vertex = gcs.AddVertex(target_mode.get_convex_set(), "target")
    target = VertexModePair(target_vertex, target_mode)

    subgraph.connect_with_continuity_constraints(3, source)
    subgraph.connect_with_continuity_constraints(1, target)

    options = opt.GraphOfConvexSetsOptions()
    options.solver_options = SolverOptions()
    # options.solver_options.SetOption(CommonSolverOption.kPrintToConsole, 1)  # type: ignore
    options.convex_relaxation = True
    options.preprocessing = True
    options.max_rounded_paths = 1

    # target_vertex = subgraph.non_collision_vertices[
    #     2
    # ]  # fails for 1 and 2 for some reason?
    result = gcs.SolveShortestPath(source_vertex, target_vertex, options)
    assert result.is_success()

    pairs = subgraph.get_all_vertex_mode_pairs()
    pairs["source"] = source
    pairs["target"] = target

    traj = PlanarTrajectoryBuilder.from_result(
        result, gcs, source_vertex, target_vertex, pairs
    ).get_trajectory(interpolate=True)
    # breakpoint()

    # p_c_W_initial = slider_initial_pose.pos() +
    # assert np.allclose(traj.p_c_W[:, 0:1], p_c_W_initial)

    DEBUG = False
    if DEBUG:
        save_gcs_graph_diagram(gcs, Path("subgraph_w_contact.svg"))
        save_gcs_graph_diagram(gcs, Path("subgraph_w_contact_result.svg"), result)
        visualize_planar_pushing_trajectory(traj, rigid_body_box.geometry)

    # breakpoint()


#
#
# p_c_W_final = target_mode.slider_pose.pos() + target_mode.p_BF_final
# assert np.allclose(traj.p_c_W[:, -1:], p_c_W_final)
#
# # Make sure we are not leaving the object
# assert np.all(np.abs(traj.p_c_W) <= 1.0)
#
# DEBUG = False
# if DEBUG:
#     save_gcs_graph_diagram(gcs, Path("subgraph.svg"))
#     save_gcs_graph_diagram(gcs, Path("subgraph_result.svg"), result)
#     visualize_planar_pushing_trajectory(traj, rigid_body_box.geometry)


# def test_planar_pushing_planner_make_plan(
#     planar_pushing_planner: PlanarPushingPlanner,
# ) -> None:
#     finger_initial_pose = PlanarPose(x=-0.3, y=0, theta=0.0)
#
#     box_initial_pose = PlanarPose(x=0.0, y=0.0, theta=0.0)
#     box_target_pose = PlanarPose(x=0.5, y=0.5, theta=0.0)
#
#     planar_pushing_planner.set_initial_poses(
#         finger_initial_pose.pos(),
#         box_initial_pose,
#         PolytopeContactLocation(ContactLocation.FACE, 3),
#     )
#
#     planar_pushing_planner.set_target_poses(
#         finger_initial_pose.pos(),
#         box_target_pose,
#         PolytopeContactLocation(ContactLocation.FACE, 3),
#     )
#
#     traj = planar_pushing_planner.make_trajectory(
#         interpolate=False, print_path=True, measure_time=True, print_output=True
#     )
#
#     DEBUG = True
#     if DEBUG:
#         save_gcs_graph_diagram(
#             planar_pushing_planner.gcs, Path("planar_pushing_graph.svg")
#         )


if __name__ == "__main__":
    # test_planar_pushing_planner_construction(
    #     planar_pushing_planner(rigid_body_box(box_geometry()))
    # )
    #
    # test_planar_pushing_planner_set_initial_and_final(
    #     planar_pushing_planner(rigid_body_box(box_geometry()))
    # )

    # test_planar_pushing_planner_make_plan(
    #     planar_pushing_planner(rigid_body_box(box_geometry()))
    # )

    test_subgraph_with_contact_modes(rigid_body_box(box_geometry()))
