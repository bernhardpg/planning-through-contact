from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest
from _pytest.fixtures import FixtureRequest
from pydrake.solvers import LinearCost, MathematicalProgramResult
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
from tests.geometry.planar.fixtures import box_geometry, planner, rigid_body_box
from tests.geometry.planar.tools import (
    assert_initial_and_final_poses,
    assert_planning_path_matches_target,
)


@pytest.mark.parametrize("planner", [{"partial": False}], indirect=["planner"])
def test_planar_pushing_planner_construction(
    planner: PlanarPushingPlanner,
) -> None:
    # One contact mode per face
    assert len(planner.contact_modes) == 4

    # One contact mode per face
    assert len(planner.contact_vertices) == 4

    # One subgraph between each contact mode:
    # 4 choose 2 = 6
    assert len(planner.subgraphs) == 6

    for v, m in zip(planner.contact_vertices, planner.contact_modes):
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

    assert planner.source_subgraph is not None
    assert planner.target_subgraph is not None

    DEBUG = False
    if DEBUG:
        save_gcs_graph_diagram(planner.gcs, Path("planar_pushing_graph.svg"))


@pytest.mark.parametrize(
    "planner", [{"partial": False, "initial_conds": True}], indirect=["planner"]
)
def test_planar_pushing_planner_set_initial_and_final(
    planner: PlanarPushingPlanner,
) -> None:
    DEBUG = False
    if DEBUG:
        save_gcs_graph_diagram(planner.gcs, Path("planar_pushing_graph.svg"))

    num_vertices_per_subgraph = 4
    num_contact_modes = 4
    num_subgraphs = 8  # 4 choose 2 + entry and exit
    expected_num_vertices = (
        num_contact_modes
        + num_vertices_per_subgraph * num_subgraphs
        + 2  # source and target vertices
    )
    assert len(planner.gcs.Vertices()) == expected_num_vertices

    num_edges_per_subgraph = 8
    num_edges_per_contact_mode = 8  # 2 to all three other modes + entry and exit
    expected_num_edges = (
        num_edges_per_subgraph * num_subgraphs
        + num_edges_per_contact_mode * num_contact_modes
        + 2  # source and target
    )
    assert len(planner.gcs.Edges()) == expected_num_edges


@pytest.mark.parametrize(
    "planner, source_idx, target_idx, target_path",
    [
        (
            {"partial": True, "initial_conds": False},
            0,
            1,
            [
                "FACE_0",
                "FACE_0_to_FACE_1_NON_COLL_0",
                "FACE_0_to_FACE_1_NON_COLL_1",
                "FACE_1",
            ],
        )
    ],
    indirect=["planner"],
)
def test_planner_wo_boundary_conds_with_contact_mode(
    planner: PlanarPushingPlanner,
    source_idx: int,
    target_idx: int,
    target_path: List[str],
) -> None:
    planner.source = VertexModePair(
        planner.contact_vertices[source_idx],
        planner.contact_modes[source_idx],
    )
    planner.target = VertexModePair(
        planner.contact_vertices[target_idx],
        planner.contact_modes[target_idx],
    )

    result = planner._solve()
    # should find a solution when there are no initial conditions
    assert result.is_success()

    assert_planning_path_matches_target(planner, result, target_path)

    DEBUG = False
    if DEBUG:
        save_gcs_graph_diagram(planner.gcs, Path("planar_pushing_graph.svg"))


@pytest.mark.parametrize(
    "planner, source_idx, target_idx, target_path",
    [
        (
            {"partial": True, "initial_conds": False},
            2,
            3,
            [
                "ENTRY_NON_COLL_2",
                "ENTRY_NON_COLL_1",
                "FACE_1",
                "EXIT_NON_COLL_1",
                "EXIT_NON_COLL_2",
                "EXIT_NON_COLL_3",
            ],
        )
    ],
    indirect=["planner"],
)
def test_planner_wo_boundary_conds_with_non_collision_mode(
    planner: PlanarPushingPlanner,
    source_idx: int,
    target_idx: int,
    target_path: List[str],
) -> None:
    planner.source = VertexModePair(
        planner.source_subgraph.non_collision_vertices[source_idx],
        planner.source_subgraph.non_collision_modes[source_idx],
    )
    planner.target = VertexModePair(
        planner.target_subgraph.non_collision_vertices[target_idx],
        planner.target_subgraph.non_collision_modes[target_idx],
    )

    result = planner._solve()
    assert result.is_success()

    assert_planning_path_matches_target(planner, result, target_path)

    DEBUG = False
    if DEBUG:
        save_gcs_graph_diagram(planner.gcs, Path("planar_pushing_graph.svg"))


@pytest.mark.parametrize(
    "planner, initial_poses, target_path",
    [
        (
            {"partial": True},
            {
                "finger_initial_pose": PlanarPose(x=0, y=-0.5, theta=0.0),
                "finger_target_pose": PlanarPose(x=-0.3, y=0, theta=0.0),
                "box_initial_pose": PlanarPose(x=0, y=0, theta=0.0),
                "box_target_pose": PlanarPose(x=-0.2, y=-0.2, theta=0.4),
            },
            [
                "source",
                "ENTRY_NON_COLL_2",
                "ENTRY_NON_COLL_1",
                "FACE_1",
                "FACE_0_to_FACE_1_NON_COLL_1",
                "FACE_0_to_FACE_1_NON_COLL_0",
                "FACE_0",
                "EXIT_NON_COLL_0",
                "EXIT_NON_COLL_3",
                "target",
            ],
        )
    ],
    indirect=["planner"],
)
def test_make_plan(
    planner: PlanarPushingPlanner,
    initial_poses: Dict[str, PlanarPose],
    target_path: List[str],
) -> None:
    planner.set_initial_poses(
        initial_poses["finger_initial_pose"],
        initial_poses["box_initial_pose"],
    )
    planner.set_target_poses(
        initial_poses["finger_target_pose"],
        initial_poses["box_target_pose"],
    )

    result = planner._solve()
    assert result.is_success()

    assert_planning_path_matches_target(planner, result, target_path)

    path = planner._get_gcs_solution_path(result)
    traj = PlanarTrajectoryBuilder(path).get_trajectory(interpolate=True)

    assert_initial_and_final_poses(
        traj,
        initial_poses["box_initial_pose"],
        initial_poses["finger_initial_pose"],
        initial_poses["box_target_pose"],
        initial_poses["finger_target_pose"],
    )

    # Make sure we are not leaving the object
    assert np.all(np.abs(traj.p_c_W) <= 1.0)

    DEBUG = False
    if DEBUG:
        save_gcs_graph_diagram(planner.gcs, Path("planar_pushing_graph.svg"))
        visualize_planar_pushing_trajectory(traj, planner.slider.geometry)
