from pathlib import Path

import numpy as np
import pytest
from pydrake.solvers import LinearCost
from pydrake.symbolic import Variables

from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    ContactLocation,
    PolytopeContactLocation,
)
from planning_through_contact.geometry.planar.non_collision_subgraph import (
    VertexModePair,
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


@pytest.fixture
def partial_planar_pushing_planner(rigid_body_box: RigidBody) -> PlanarPushingPlanner:
    """
    Planar pushing planner that does not consider all the contact locations, in order
    for tests to be a bit quicker
    """
    specs = PlanarPlanSpecs()
    contact_locations = rigid_body_box.geometry.contact_locations[0:2]
    return PlanarPushingPlanner(
        rigid_body_box, specs, contact_locations=contact_locations
    )


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
    )

    planar_pushing_planner.set_target_poses(
        finger_initial_pose.pos(),
        box_target_pose,
    )

    DEBUG = False
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


def test_planar_pushing_planner_without_initial_conds(
    partial_planar_pushing_planner: PlanarPushingPlanner,
) -> None:
    planner = partial_planar_pushing_planner
    planner.source = VertexModePair(
        planner.contact_vertices[0],
        planner.contact_modes[0],
    )
    planner.target = VertexModePair(
        planner.contact_vertices[1],
        planner.contact_modes[1],
    )

    result = planner._solve()
    # should find a solution when there are no initial conditions
    assert result.is_success()

    vertex_path = planner.get_vertex_solution_path(result)
    target_path = [
        "FACE_0",
        "FACE_0_to_FACE_1_NON_COLL_0",
        "FACE_0_to_FACE_1_NON_COLL_1",
        "FACE_1",
    ]
    for v, target in zip(vertex_path, target_path):
        assert v.name() == target

    DEBUG = False
    if DEBUG:
        save_gcs_graph_diagram(planner.gcs, Path("planar_pushing_graph.svg"))


def test_planar_pushing_planner_without_initial_conds_2(
    partial_planar_pushing_planner: PlanarPushingPlanner,
) -> None:
    planner = partial_planar_pushing_planner
    planner.source = VertexModePair(
        planner.source_subgraph.non_collision_vertices[2],
        planner.source_subgraph.non_collision_modes[2],
    )
    planner.target = VertexModePair(
        planner.target_subgraph.non_collision_vertices[3],
        planner.target_subgraph.non_collision_modes[3],
    )

    result = planner._solve()
    assert result.is_success()

    vertex_path = planner.get_vertex_solution_path(result)
    target_path = [
        "ENTRY_NON_COLL_2",
        "ENTRY_NON_COLL_1",
        "FACE_1",
        "EXIT_NON_COLL_1",
        "EXIT_NON_COLL_2",
        "EXIT_NON_COLL_3",
    ]
    for v, target in zip(vertex_path, target_path):
        assert v.name() == target

    DEBUG = False
    if DEBUG:
        save_gcs_graph_diagram(planner.gcs, Path("planar_pushing_graph.svg"))


def test_planar_pushing_planner_make_plan(
    partial_planar_pushing_planner: PlanarPushingPlanner,
) -> None:
    planner = partial_planar_pushing_planner
    finger_initial_pose = PlanarPose(x=0.5, y=0, theta=0.0)
    finger_target_pose = PlanarPose(x=0.5, y=-0.2, theta=0.0)

    box_initial_pose = PlanarPose(x=0.0, y=0.0, theta=0.0)
    box_target_pose = PlanarPose(x=-0.2, y=-0.2, theta=0.4)

    planner.set_initial_poses(
        finger_initial_pose.pos(),
        box_initial_pose,
    )
    planner.set_target_poses(
        finger_target_pose.pos(),
        box_target_pose,
    )

    result = planner._solve()
    assert result.is_success()

    vertex_path = planner.get_vertex_solution_path(result)
    target_path = [
        "source",
        "ENTRY_NON_COLL_1",
        "ENTRY_NON_COLL_0",
        "FACE_0",
        "FACE_0_to_FACE_1_NON_COLL_0",
        "FACE_0_to_FACE_1_NON_COLL_1",
        "FACE_1",
        "EXIT_NON_COLL_1",
        "target",
    ]
    for v, target in zip(vertex_path, target_path):
        assert v.name() == target

    path = planner._get_gcs_solution_path(result)
    traj = PlanarTrajectoryBuilder(path).get_trajectory(interpolate=True)

    p_c_W_initial = box_initial_pose.pos() + finger_initial_pose.pos()
    assert np.allclose(traj.p_c_W[:, 0:1], p_c_W_initial)

    p_c_W_final = box_target_pose.pos() + box_target_pose.two_d_rot_matrix().dot(
        finger_target_pose.pos()
    )
    assert np.allclose(traj.p_c_W[:, -1:], p_c_W_final)

    # Make sure we are not leaving the object
    assert np.all(np.abs(traj.p_c_W) <= 1.0)

    DEBUG = False
    if DEBUG:
        save_gcs_graph_diagram(planner.gcs, Path("planar_pushing_graph.svg"))
        visualize_planar_pushing_trajectory(traj, planner.slider.geometry)
