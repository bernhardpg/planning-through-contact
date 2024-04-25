import os
from pathlib import Path

import numpy as np
import pytest
from pydrake.solvers import LinearCost
from pydrake.symbolic import Variables

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
from planning_through_contact.planning.planar.planar_plan_config import (
    PlanarSolverParams,
)
from planning_through_contact.planning.planar.planar_pushing_planner import (
    PlanarPushingPlanner,
)
from planning_through_contact.visualize.analysis import save_gcs_graph_diagram
from planning_through_contact.visualize.planar_pushing import (
    make_traj_figure,
    visualize_planar_pushing_trajectory,
    visualize_planar_pushing_trajectory_legacy,
)
from tests.geometry.planar.fixtures import (
    box_geometry,
    dynamics_config,
    plan_config,
    planner,
    rigid_body_box,
    t_pusher,
)
from tests.geometry.planar.tools import assert_initial_and_final_poses

DEBUG = False
IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


@pytest.mark.parametrize(
    "planner",
    [
        {
            "partial": False,
            "boundary_conds": {
                "finger_initial_pose": PlanarPose(x=0, y=-0.4, theta=0.0),
                "finger_target_pose": PlanarPose(x=-0.3, y=0, theta=0.0),
                "box_initial_pose": PlanarPose(x=0, y=0, theta=0.0),
                "box_target_pose": PlanarPose(x=-0.2, y=-0.2, theta=0.4),
            },
        }
    ],
    indirect=["planner"],
)
def test_planner_construction(
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

        # (angular velocity, linear velocity, normal force, friction force) * one term per (knot point - 1)
        assert len(costs) == len(m.prog.GetAllCosts())

        for idx, cost in enumerate(costs):
            if len(m.prog.linear_costs()) > 0:
                var_idxs = m._get_cost_terms(m.prog.linear_costs())[0][idx]
                target_vars = Variables(v.x()[var_idxs])
                assert target_vars.EqualTo(Variables(cost.variables()))

                # Costs should be linear in SDP relaxation
                assert isinstance(cost.evaluator(), LinearCost)

    assert planner.source_subgraph is not None
    assert planner.target_subgraph is not None

    if DEBUG:
        save_gcs_graph_diagram(planner.gcs, Path("planar_pushing_graph.svg"))


@pytest.mark.parametrize(
    "planner",
    [
        {
            "partial": False,
            "allow_teleportation": False,
            "boundary_conds": {
                "finger_initial_pose": PlanarPose(x=0, y=-0.4, theta=0.0),
                "finger_target_pose": PlanarPose(x=-0.3, y=0, theta=0.0),
                "box_initial_pose": PlanarPose(x=0, y=0, theta=0.0),
                "box_target_pose": PlanarPose(x=-0.2, y=-0.2, theta=0.4),
            },
        }
    ],
    indirect=["planner"],
)
def test_planner_set_initial_and_final(
    planner: PlanarPushingPlanner,
) -> None:
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
    "planner",
    [
        {
            "partial": False,
            "allow_teleportation": True,
            "boundary_conds": {
                "finger_initial_pose": PlanarPose(x=0, y=-0.4, theta=0.0),
                "finger_target_pose": PlanarPose(x=-0.3, y=0, theta=0.0),
                "box_initial_pose": PlanarPose(x=0, y=0, theta=0.0),
                "box_target_pose": PlanarPose(x=-0.2, y=-0.2, theta=0.4),
            },
        }
    ],
    indirect=["planner"],
)
def test_planner_construction_with_teleportation(planner: PlanarPushingPlanner) -> None:
    if DEBUG:
        save_gcs_graph_diagram(planner.gcs, Path("teleportation_construction.svg"))

    num_contact_modes = 4
    if planner.source is not None and planner.target is not None:
        expected_num_vertices = num_contact_modes + 2  # source and target vertices
    else:
        expected_num_vertices = num_contact_modes

    assert len(planner.gcs.Vertices()) == expected_num_vertices

    edges_between_contact_modes = 6 * 2  # 4 nCr 2 and bi-directional edges

    if planner.source is not None and planner.target is not None:
        num_edges_from_target_and_source = num_contact_modes * 2
        expected_num_edges = (
            edges_between_contact_modes + num_edges_from_target_and_source
        )
    else:
        expected_num_edges = edges_between_contact_modes

    assert len(planner.gcs.Edges()) == expected_num_edges


@pytest.mark.parametrize(
    "planner",
    [
        (
            {
                "partial": True,
                "allow_teleportation": True,
                "penalize_mode_transition": False,
                "boundary_conds": {
                    "finger_initial_pose": PlanarPose(x=0, y=0.4, theta=0.0),
                    "finger_target_pose": PlanarPose(x=0.3, y=0, theta=0.0),
                    "box_initial_pose": PlanarPose(x=0, y=0, theta=0.0),
                    "box_target_pose": PlanarPose(x=-0.2, y=-0.2, theta=0.4),
                },
            }
        ),
        # NOTE: This test takes a few minutes, and is hence commented out
        # (
        #     {
        #         "partial": False,
        #         "allow_teleportation": True,
        #         "penalize_mode_transition": True,
        #         "boundary_conds": {
        #             "finger_initial_pose": PlanarPose(x=-0.4, y=0.0, theta=0.0),
        #             "finger_target_pose": PlanarPose(x=-0.4, y=0, theta=0.0),
        #             "box_initial_pose": PlanarPose(x=0, y=0, theta=0.0),
        #             "box_target_pose": PlanarPose(x=-0.2, y=-0.2, theta=1.1),
        #         },
        #     },
        #     ["source", "FACE_1", "target"],
        # ),
    ],
    indirect=["planner"],
)
def test_planner_with_teleportation(planner: PlanarPushingPlanner) -> None:
    if DEBUG:
        save_gcs_graph_diagram(planner.gcs, Path("teleportation_graph.svg"))
    solver_params = PlanarSolverParams(print_solver_output=DEBUG)
    result = planner._solve(solver_params)
    assert result.is_success()

    path = planner.plan_path(solver_params)
    traj = path.to_traj()
    assert_initial_and_final_poses(
        traj,
        planner.slider_pose_initial,
        planner.pusher_pose_initial,
        planner.slider_pose_target,
        planner.pusher_pose_target,
    )

    # Make sure we are not leaving the object
    assert np.all(
        [
            np.abs(p_BP) <= 1.0
            for knot_point in traj.path_knot_points
            for p_BP in knot_point.p_BPs  # type: ignore
        ]
    )

    if DEBUG:
        save_gcs_graph_diagram(planner.gcs, Path("teleportation_graph.svg"))
        visualize_planar_pushing_trajectory(
            traj, visualize_knot_points=True, save=True, filename="debug_file"
        )


@pytest.mark.skipif(
    IN_GITHUB_ACTIONS == True,
    reason="Too slow",
)
@pytest.mark.parametrize(
    "planner",
    [
        {
            "partial": True,
            "avoid_object": False,
            "boundary_conds": {
                "finger_initial_pose": PlanarPose(x=0, y=-0.3, theta=0.0),
                "finger_target_pose": PlanarPose(x=-0.3, y=0, theta=0.0),
                "box_initial_pose": PlanarPose(x=0, y=0, theta=0.0),
                "box_target_pose": PlanarPose(x=-0.2, y=-0.2, theta=0.4),
            },
        },
        {
            "partial": True,
            "avoidance_cost_type": "quadratic",
            "avoid_object": True,
            "boundary_conds": {
                "finger_initial_pose": PlanarPose(x=0, y=-0.3, theta=0.0),
                "finger_target_pose": PlanarPose(x=-0.3, y=0, theta=0.0),
                "box_initial_pose": PlanarPose(x=0.0, y=0.0, theta=0.0),
                "box_target_pose": PlanarPose(x=-0.2, y=-0.2, theta=0.4),
            },
        },
        {
            "partial": True,
            "avoidance_cost_type": "quadratic",
            "avoid_object": True,
            "boundary_conds": {
                "finger_initial_pose": PlanarPose(x=0, y=-0.3, theta=0.0),
                "finger_target_pose": PlanarPose(x=-0.3, y=0, theta=0.0),
                "box_initial_pose": PlanarPose(x=0.0, y=0.0, theta=0.0),
                "box_target_pose": PlanarPose(x=-0.2, y=-0.2, theta=0.4),
            },
            "body": "t_pusher",
        },
    ],
    indirect=["planner"],
    ids=[
        "box_collision",
        "box_non_collision",
        "t_pusher",
    ],
)
def test_make_plan(
    planner: PlanarPushingPlanner,
) -> None:
    solver_params = PlanarSolverParams(print_solver_output=DEBUG)
    result = planner._solve(solver_params)
    assert result.is_success()

    path = planner.plan_path(solver_params)
    traj = path.to_traj()
    assert_initial_and_final_poses(
        traj,
        planner.slider_pose_initial,
        planner.pusher_pose_initial,
        planner.slider_pose_target,
        planner.pusher_pose_target,
    )

    # Make sure we are not leaving the object
    assert np.all(
        [
            np.abs(p_BP) <= 1.0
            for knot_point in traj.path_knot_points
            for p_BP in knot_point.p_BPs  # type: ignore
        ]
    )

    if DEBUG:
        save_gcs_graph_diagram(planner.gcs, Path("planar_pushing_graph.svg"))
        visualize_planar_pushing_trajectory(
            traj, visualize_knot_points=True, save=True, filename="debug_file"
        )


@pytest.mark.parametrize(
    "planner",
    [
        {
            "avoid_object": True,
            "boundary_conds": {
                "finger_initial_pose": PlanarPose(x=0, y=-0.3, theta=0.0),
                "finger_target_pose": PlanarPose(x=-0.3, y=0, theta=0.0),
                "box_initial_pose": PlanarPose(x=0, y=0, theta=0.0),
                "box_target_pose": PlanarPose(x=-0.2, y=-0.2, theta=0.4),
            },
        },
        {
            "avoid_object": True,
            "boundary_conds": {
                "finger_initial_pose": PlanarPose(x=0, y=-0.3, theta=0.0),
                "finger_target_pose": PlanarPose(x=-0.3, y=0, theta=0.0),
                "box_initial_pose": PlanarPose(x=0.2, y=0.2, theta=0.5),
                "box_target_pose": PlanarPose(x=0.4, y=-0.2, theta=-0.4),
            },
        },
        {
            "avoid_object": True,
            "boundary_conds": {
                "finger_initial_pose": PlanarPose(x=0, y=-0.3, theta=0.0),
                "finger_target_pose": PlanarPose(x=-0.3, y=0, theta=0.0),
                "box_initial_pose": PlanarPose(x=0.2, y=0.2, theta=0.5),
                "box_target_pose": PlanarPose(x=0.4, y=-0.2, theta=-0.4),
            },
            "pusher_velocity_continuity": True,
        },
    ],
    indirect=["planner"],
    ids=[
        "plan_1",
        "plan_2",
        "velocity_cont",  # NOTE: This plan does not look good. The planner config should be updated in the tests
    ],
)
def test_make_plan_band_sparsity_box(
    planner: PlanarPushingPlanner,
) -> None:
    solver_params = PlanarSolverParams(print_solver_output=DEBUG)
    result = planner._solve(solver_params)
    assert result.is_success()

    path = planner.plan_path(solver_params)
    traj = path.to_traj()

    assert_initial_and_final_poses(
        traj,
        planner.slider_pose_initial,
        planner.pusher_pose_initial,
        planner.slider_pose_target,
        planner.pusher_pose_target,
    )

    # Make sure we are not leaving the object
    assert np.all(
        [
            np.abs(p_BP) <= 5.0
            for knot_point in traj.path_knot_points
            for p_BP in knot_point.p_BPs  # type: ignore
        ]
    )

    if DEBUG:
        save_gcs_graph_diagram(planner.gcs, Path("planar_pushing_graph.svg"))
        visualize_planar_pushing_trajectory(
            traj, visualize_knot_points=True, save=True, filename="debug_file"
        )


@pytest.mark.skip(reason="Not working yet due to UNKNOWN mosek error")
@pytest.mark.parametrize(
    "planner",
    [
        {
            "avoid_object": True,
            "boundary_conds": {
                "finger_initial_pose": PlanarPose(x=0, y=-0.3, theta=0.0),
                "finger_target_pose": PlanarPose(x=-0.3, y=0, theta=0.0),
                "box_initial_pose": PlanarPose(x=0, y=0, theta=0.0),
                "box_target_pose": PlanarPose(x=-0.2, y=-0.2, theta=0.4),
            },
            "body": "t_pusher",
        },
        {
            "avoid_object": True,
            "boundary_conds": {
                "finger_initial_pose": PlanarPose(x=0, y=-0.3, theta=0.0),
                "finger_target_pose": PlanarPose(x=-0.3, y=0, theta=0.0),
                "box_initial_pose": PlanarPose(x=0.2, y=0.2, theta=0.5),
                "box_target_pose": PlanarPose(x=0.4, y=-0.2, theta=-0.4),
            },
            "body": "t_pusher",
        },
    ],
    indirect=["planner"],
    ids=[1, 2],
)
def test_make_plan_band_sparsity_t_pusher(
    planner: PlanarPushingPlanner,
) -> None:
    solver_params = PlanarSolverParams(save_solver_output=DEBUG)
    result = planner._solve(solver_params)
    assert result.is_success()

    path = planner.plan_path(solver_params)
    traj = path.to_traj()

    assert_initial_and_final_poses(
        traj,
        planner.slider_pose_initial,
        planner.pusher_pose_initial,
        planner.slider_pose_target,
        planner.pusher_pose_target,
    )

    # Make sure we are not leaving the object
    assert np.all(
        [
            np.abs(p_BP) <= 1.0
            for knot_point in traj.path_knot_points
            for p_BP in knot_point.p_BPs  # type: ignore
        ]
    )

    if DEBUG:
        save_gcs_graph_diagram(planner.gcs, Path("planar_pushing_graph.svg"))
        visualize_planar_pushing_trajectory(
            traj, visualize_knot_points=True, save=True, filename="debug_file"
        )


@pytest.mark.parametrize(
    "planner",
    [
        {
            "avoid_object": True,
            "boundary_conds": {
                "finger_initial_pose": PlanarPose(x=0, y=-0.3, theta=0.0),
                "finger_target_pose": PlanarPose(x=-0.3, y=0, theta=0.0),
                "box_initial_pose": PlanarPose(x=0, y=0, theta=0.0),
                "box_target_pose": PlanarPose(x=-0.2, y=-0.2, theta=0.4),
            },
            "body": "box",
            "standard_cost": True,
        },
    ],
    indirect=["planner"],
    ids=[1],
)
def test_planner_standard_cost(
    planner: PlanarPushingPlanner,
) -> None:
    solver_params = PlanarSolverParams(save_solver_output=DEBUG)
    result = planner._solve(solver_params)
    assert result.is_success()

    path = planner.plan_path(solver_params)
    traj = path.to_traj()

    assert_initial_and_final_poses(
        traj,
        planner.slider_pose_initial,
        planner.pusher_pose_initial,
        planner.slider_pose_target,
        planner.pusher_pose_target,
    )

    # Make sure we are not leaving the object
    assert np.all(
        [
            np.abs(p_BP) <= 1.0
            for knot_point in traj.path_knot_points
            for p_BP in knot_point.p_BPs  # type: ignore
        ]
    )

    if DEBUG:
        save_gcs_graph_diagram(planner.gcs, Path("planar_pushing_graph.svg"))
        visualize_planar_pushing_trajectory(
            traj, visualize_knot_points=True, save=True, filename="debug_file"
        )


@pytest.mark.parametrize(
    "planner",
    [
        {
            "avoid_object": True,
            "boundary_conds": {
                "finger_initial_pose": PlanarPose(x=0, y=-0.3, theta=0.0),
                "finger_target_pose": PlanarPose(x=-0.3, y=0, theta=0.0),
                "box_initial_pose": PlanarPose(x=0, y=0, theta=0.0),
                "box_target_pose": PlanarPose(x=-0.2, y=-0.2, theta=0.4),
            },
            "body": "box",
            "standard_cost": True,
        },
    ],
    indirect=["planner"],
    ids=[1],
)
def test_planner_plan_path(
    planner: PlanarPushingPlanner,
) -> None:
    solver_params = PlanarSolverParams(save_solver_output=DEBUG)
    path = planner.plan_path(solver_params)
    traj = path.to_traj(rounded=True)

    assert_initial_and_final_poses(
        traj,
        planner.slider_pose_initial,
        planner.pusher_pose_initial,
        planner.slider_pose_target,
        planner.pusher_pose_target,
    )

    # Make sure we are not leaving the object
    assert np.all(
        [
            np.abs(p_BP) <= 1.0
            for knot_point in traj.path_knot_points
            for p_BP in knot_point.p_BPs  # type: ignore
        ]
    )

    if DEBUG:
        save_gcs_graph_diagram(planner.gcs, Path("planar_pushing_graph.svg"))
        visualize_planar_pushing_trajectory(
            traj, visualize_knot_points=True, save=True, filename="debug_file"
        )
        make_traj_figure(
            traj, plot_lims=(-0.5, 0.1, -0.5, 0.1), filename="debug_file_traj"
        )
