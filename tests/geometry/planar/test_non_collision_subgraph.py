from pathlib import Path

import numpy as np
import pydrake.geometry.optimization as opt
import pytest
from pydrake.solvers import LinearCost, MosekSolver, QuadraticCost

from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    ContactLocation,
    PolytopeContactLocation,
)
from planning_through_contact.geometry.collision_geometry.t_pusher_2d import TPusher2d
from planning_through_contact.geometry.planar.face_contact import FaceContactMode
from planning_through_contact.geometry.planar.non_collision import (
    NonCollisionMode,
    find_first_matching_location,
)
from planning_through_contact.geometry.planar.non_collision_subgraph import (
    NonCollisionSubGraph,
    VertexModePair,
)
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.planar.planar_pushing_trajectory import (
    PlanarPushingTrajectory,
)
from planning_through_contact.geometry.planar.trajectory_builder import (
    PlanarTrajectoryBuilder,
)
from planning_through_contact.geometry.rigid_body import RigidBody
from planning_through_contact.planning.planar.planar_plan_config import PlanarPlanConfig
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
    subgraph,
)
from tests.geometry.planar.tools import (
    assert_initial_and_final_poses,
    assert_initial_and_final_poses_LEGACY,
    assert_object_is_avoided,
)

DEBUG = False
# TODO(bernhardpg): Old visualizer is no longer working after an update to the integration scheme, replace all old visualizer instances
# with the new visualizer!


@pytest.mark.parametrize(
    "subgraph",
    [
        {"boundary_conds": False, "avoid_object": False},
        {"boundary_conds": False, "avoid_object": True},
    ],
    indirect=["subgraph"],
)
def test_non_collision_subgraph(subgraph: NonCollisionSubGraph):
    assert len(subgraph.gcs.Vertices()) == len(subgraph.non_collision_modes)

    # Edges are bi-directional, one edge between each mode for box
    assert len(subgraph.gcs.Edges()) == len(subgraph.non_collision_modes) * 2

    num_continuity_variables = 6
    for edge in subgraph.gcs.Edges():
        assert len(edge.GetConstraints()) == num_continuity_variables

    if subgraph.config.avoid_object and subgraph.config.avoidance_cost == "quadratic":
        # Check costs are correctly added to GCS instance
        for v in subgraph.gcs.Vertices():
            if v.name() in ("source", "target"):
                continue

            costs = v.GetCosts()
            assert len(costs) == 2

            # eucl distance cost
            assert isinstance(costs[0].evaluator(), QuadraticCost)

            # maximize distance cost
            assert isinstance(costs[1].evaluator(), QuadraticCost)

    else:
        for vertex in subgraph.gcs.Vertices():
            # Squared eucl distance
            costs = vertex.GetCosts()
            assert len(costs) == 1

            # p_BF for each knot point should be in the cost
            cost = costs[0]
            NUM_DIMS = 2
            assert (
                len(cost.variables())
                == subgraph.config.num_knot_points_non_collision * NUM_DIMS
            )

            # Squared eucl distance
            assert isinstance(cost.evaluator(), QuadraticCost)


@pytest.mark.parametrize(
    "subgraph",
    [
        {
            "boundary_conds": True,
            "avoid_object": False,
            "finger_initial": PlanarPose(-0.20, 0, 0),
            "finger_final": PlanarPose(0.20, 0, 0),
        },
        {
            "boundary_conds": True,
            "avoid_object": True,
            "finger_initial": PlanarPose(-0.20, 0, 0),
            "finger_final": PlanarPose(0.20, 0, 0),
        },
    ],
    indirect=["subgraph"],
)
def test_non_collision_subgraph_initial_and_final(
    subgraph: NonCollisionSubGraph,
):
    assert subgraph.source is not None
    source_mode = subgraph.source.mode

    assert isinstance(source_mode, NonCollisionMode)

    assert source_mode.contact_location == find_first_matching_location(
        source_mode.finger_initial_pose, subgraph.config
    )
    assert subgraph.target is not None

    target_mode = subgraph.target.mode

    assert isinstance(target_mode, NonCollisionMode)

    assert target_mode.contact_location == find_first_matching_location(
        target_mode.finger_final_pose, subgraph.config
    )

    # We should have added 2 more edges with initial and final modes
    assert len(subgraph.gcs.Edges()) == len(subgraph.non_collision_modes) * 2 + 2


@pytest.mark.parametrize(
    "subgraph",
    [
        {
            "boundary_conds": True,
            "avoid_object": False,
            "finger_initial": PlanarPose(-0.20, 0, 0),
            "finger_final": PlanarPose(0.20, -0.1, 0),
        },
        {
            "boundary_conds": True,
            "avoid_object": False,
            "finger_initial": PlanarPose(-0.20, 0, 0),
            "finger_final": PlanarPose(0.20, 0.1, 0),
        },
        {
            "boundary_conds": True,
            "avoid_object": True,
            "finger_initial": PlanarPose(-0.20, 0, 0),
            "finger_final": PlanarPose(0.20, -0.1, 0),
        },
        {
            "boundary_conds": True,
            "avoid_object": True,
            "finger_initial": PlanarPose(-0.20, 0, 0),
            "finger_final": PlanarPose(0.20, 0.1, 0),
        },
    ],
    indirect=["subgraph"],
)
def test_subgraph_planning(
    subgraph: NonCollisionSubGraph,
):
    # get rid of all LSP errors
    assert subgraph.source is not None
    assert subgraph.target is not None

    options = opt.GraphOfConvexSetsOptions()
    options.solver = MosekSolver()
    result = subgraph.gcs.SolveShortestPath(
        subgraph.source.vertex,
        subgraph.target.vertex,
        options,
    )
    assert result.is_success()

    # TODO(bernhardpg): use new Drake function:
    # edges = gcs.GetSolutionPath(source, target, result)

    pairs = subgraph.get_all_vertex_mode_pairs()
    pairs["source"] = subgraph.source
    pairs["target"] = subgraph.target
    traj = PlanarTrajectoryBuilder.from_result(
        result, subgraph.gcs, subgraph.source.vertex, subgraph.target.vertex, pairs
    ).get_trajectory(interpolate=False)

    assert isinstance(subgraph.source.mode, NonCollisionMode)
    assert isinstance(subgraph.target.mode, NonCollisionMode)
    assert_initial_and_final_poses_LEGACY(
        traj,
        subgraph.source.mode.slider_pose,
        subgraph.source.mode.finger_initial_pose,
        subgraph.target.mode.slider_pose,
        subgraph.target.mode.finger_final_pose,
    )

    # Make sure we are not leaving the object
    assert np.all(np.abs(traj.p_BP) <= 1.0)

    # Make sure we always take the shortest path
    vertex_names = [
        v.name()
        for v in get_gcs_solution_path_vertices(
            subgraph.gcs, result, subgraph.source.vertex, subgraph.target.vertex
        )
    ]
    should_go_around_above = subgraph.target.mode.finger_final_pose.y >= 0
    if should_go_around_above:
        targets = [
            "source",
            "Subgraph_TEST_NON_COLL_3",
            "Subgraph_TEST_NON_COLL_0",
            "Subgraph_TEST_NON_COLL_1",
            "target",
        ]
    else:
        targets = [
            "source",
            "Subgraph_TEST_NON_COLL_3",
            "Subgraph_TEST_NON_COLL_2",
            "Subgraph_TEST_NON_COLL_1",
            "target",
        ]
    assert all([v == t for v, t in zip(vertex_names, targets)])

    if subgraph.config.avoid_object:
        # check that all trajectory points (after source and target modes) don't collide
        assert_object_is_avoided(
            subgraph.slider.geometry,
            traj.p_BP,
            min_distance=0.001,
            start_idx=2,
            end_idx=-2,
        )

    if DEBUG:
        save_gcs_graph_diagram(subgraph.gcs, Path("subgraph.svg"))
        save_gcs_graph_diagram(subgraph.gcs, Path("subgraph_result.svg"), result)
        visualize_planar_pushing_trajectory_legacy(traj, subgraph.slider.geometry, 0.01)


@pytest.mark.parametrize(
    "subgraph",
    [
        {
            "boundary_conds": False,
            "avoid_object": False,
        },
        {
            "boundary_conds": False,
            "avoid_object": True,
        },
    ],
    indirect=["subgraph"],
    ids=[
        "simple",
        "avoid_object",
    ],
)
def test_subgraph_with_contact_modes(
    subgraph: NonCollisionSubGraph,
    gcs_options: opt.GraphOfConvexSetsOptions,
):
    """
    This unit test tests much of the code that is implemented in the PlanarPushingPlanner.
    """

    contact_location_start = PolytopeContactLocation(ContactLocation.FACE, 3)
    source_mode = FaceContactMode.create_from_plan_spec(
        contact_location_start, subgraph.config
    )
    source_mode.set_finger_pos(0.5)

    contact_location_end = PolytopeContactLocation(ContactLocation.FACE, 2)
    target_mode = FaceContactMode.create_from_plan_spec(
        contact_location_end, subgraph.config
    )
    target_mode.set_finger_pos(0.5)

    slider_initial_pose = PlanarPose(0.3, 0, 0)
    source_mode.set_slider_initial_pose(slider_initial_pose)
    source_vertex = subgraph.gcs.AddVertex(source_mode.get_convex_set(), "source")
    source_mode.add_cost_to_vertex(source_vertex)
    source = VertexModePair(source_vertex, source_mode)

    slider_final_pose = PlanarPose(0.5, 0.3, 0.4)
    target_mode.set_slider_final_pose(slider_final_pose)
    target_vertex = subgraph.gcs.AddVertex(target_mode.get_convex_set(), "target")
    target_mode.add_cost_to_vertex(target_vertex)
    target = VertexModePair(target_vertex, target_mode)

    subgraph.connect_with_continuity_constraints(
        contact_location_start.idx, source, outgoing=False
    )
    subgraph.connect_with_continuity_constraints(
        contact_location_end.idx, target, incoming=False
    )

    result = subgraph.gcs.SolveShortestPath(source_vertex, target_vertex, gcs_options)
    assert result.is_success()

    pairs = subgraph.get_all_vertex_mode_pairs()
    pairs["source"] = source
    pairs["target"] = target

    traj = PlanarPushingTrajectory.from_result(
        subgraph.config, result, subgraph.gcs, source_vertex, target_vertex, pairs
    )

    assert_initial_and_final_poses(
        traj, slider_initial_pose, None, slider_final_pose, None
    )

    # Make sure we are not leaving the object
    assert np.all(
        [
            np.abs(p_BP) <= 1.0
            for knot_point in traj.path_knot_points
            for p_BP in knot_point.p_BPs  # type: ignore
        ]
    )

    if subgraph.config.avoid_object:
        first_segment = np.hstack(traj.path_knot_points[1].p_BPs)  # type: ignore
        assert_object_is_avoided(
            subgraph.slider.geometry,
            first_segment,
            min_distance=0.001,
            start_idx=2,
            end_idx=-2,
        )
        second_segment = np.hstack(traj.path_knot_points[2].p_BPs)  # type: ignore
        assert_object_is_avoided(
            subgraph.slider.geometry,
            second_segment,
            min_distance=0.001,
            start_idx=2,
            end_idx=-2,
        )

    if DEBUG:
        save_gcs_graph_diagram(subgraph.gcs, Path("subgraph_w_contact.svg"))
        visualize_planar_pushing_trajectory(
            traj, visualize_knot_points=True, save=True, filename="debug_file"
        )


@pytest.mark.parametrize("avoid_object", [False, True], ids=["non_avoid", "avoid"])
def test_subgraph_planning_t_pusher(plan_config: PlanarPlanConfig, avoid_object: bool):
    plan_config.avoid_object = avoid_object
    plan_config.num_knot_points_non_collision = 4
    plan_config.dynamics_config.slider = RigidBody("T", TPusher2d(), mass=0.2)
    gcs = opt.GraphOfConvexSets()

    subgraph = NonCollisionSubGraph.create_with_gcs(
        gcs,
        plan_config,
        "Subgraph_test_T",
    )

    slider_pose = PlanarPose(0.0, 0.0, 0)
    initial = PlanarPose(-0.20, 0, 0)
    target = PlanarPose(0.20, -0.1, 0)

    subgraph.set_initial_poses(initial, slider_pose)
    subgraph.set_final_poses(target, slider_pose)

    # get rid of all LSP errors
    assert subgraph.source is not None
    assert subgraph.target is not None

    options = opt.GraphOfConvexSetsOptions()
    options.solver = MosekSolver()
    result = subgraph.gcs.SolveShortestPath(
        subgraph.source.vertex, subgraph.target.vertex, options
    )
    assert result.is_success()

    pairs = subgraph.get_all_vertex_mode_pairs()
    pairs["source"] = subgraph.source
    pairs["target"] = subgraph.target
    traj = PlanarTrajectoryBuilder.from_result(
        result, subgraph.gcs, subgraph.source.vertex, subgraph.target.vertex, pairs
    ).get_trajectory(interpolate=False)

    assert isinstance(subgraph.source.mode, NonCollisionMode)
    assert isinstance(subgraph.target.mode, NonCollisionMode)
    assert_initial_and_final_poses_LEGACY(
        traj,
        subgraph.source.mode.slider_pose,
        subgraph.source.mode.finger_initial_pose,
        subgraph.target.mode.slider_pose,
        subgraph.target.mode.finger_final_pose,
    )

    # Make sure we are not leaving the object
    assert np.all(np.abs(traj.p_BP) <= 1.0)

    if subgraph.config.avoid_object:
        # check that all trajectory points (after source and target modes) don't collide
        assert_object_is_avoided(
            subgraph.slider.geometry,
            traj.p_BP,
            min_distance=0.001,
            start_idx=2,
            end_idx=-2,
        )

    if DEBUG:
        save_gcs_graph_diagram(subgraph.gcs, Path("subgraph.svg"))
        save_gcs_graph_diagram(subgraph.gcs, Path("subgraph_result.svg"), result)
        visualize_planar_pushing_trajectory_legacy(traj, subgraph.slider.geometry, 0.01)


@pytest.mark.parametrize("avoid_object", [False, True], ids=["non_avoid", "avoid"])
def test_subgraph_contact_modes_t_pusher(
    plan_config: PlanarPlanConfig, avoid_object: bool, gcs_options
):
    plan_config.avoid_object = avoid_object
    plan_config.num_knot_points_non_collision = 4
    plan_config.dynamics_config.slider = RigidBody("T", TPusher2d(), mass=0.2)
    plan_config.dynamics_config.pusher_radius = 0.015
    gcs = opt.GraphOfConvexSets()

    subgraph = NonCollisionSubGraph.create_with_gcs(
        gcs,
        plan_config,
        "Subgraph_test_T",
    )

    contact_location_start = PolytopeContactLocation(ContactLocation.FACE, 1)
    source_mode = FaceContactMode.create_from_plan_spec(
        contact_location_start, subgraph.config
    )
    source_mode.set_finger_pos(0.5)

    contact_location_end = PolytopeContactLocation(ContactLocation.FACE, 4)
    target_mode = FaceContactMode.create_from_plan_spec(
        contact_location_end, subgraph.config
    )
    target_mode.set_finger_pos(0.5)

    slider_initial_pose = PlanarPose(0.3, 0, 0)
    source_mode.set_slider_initial_pose(slider_initial_pose)
    source_vertex = subgraph.gcs.AddVertex(source_mode.get_convex_set(), "FACE_1")
    source_mode.add_cost_to_vertex(source_vertex)
    source = VertexModePair(source_vertex, source_mode)

    slider_final_pose = PlanarPose(-0.5, 0.3, 0.4)
    target_mode.set_slider_final_pose(slider_final_pose)
    target_vertex = subgraph.gcs.AddVertex(target_mode.get_convex_set(), "FACE_4")
    target_mode.add_cost_to_vertex(target_vertex)
    target = VertexModePair(target_vertex, target_mode)

    subgraph.connect_with_continuity_constraints(
        plan_config.slider_geometry.get_collision_free_region_for_loc_idx(
            contact_location_start.idx
        ),
        source,
        outgoing=False,
    )
    subgraph.connect_with_continuity_constraints(
        plan_config.slider_geometry.get_collision_free_region_for_loc_idx(
            contact_location_end.idx,
        ),
        target,
        incoming=False,
    )

    save_gcs_graph_diagram(subgraph.gcs, Path("subgraph_w_contact_t_pusher.svg"))

    result = subgraph.gcs.SolveShortestPath(source_vertex, target_vertex, gcs_options)
    assert result.is_success()

    pairs = subgraph.get_all_vertex_mode_pairs()
    pairs["FACE_1"] = source
    pairs["FACE_4"] = target

    traj = PlanarTrajectoryBuilder.from_result(
        result, subgraph.gcs, source_vertex, target_vertex, pairs
    ).get_trajectory(interpolate=False)

    assert_initial_and_final_poses_LEGACY(
        traj, slider_initial_pose, None, slider_final_pose, None
    )

    if DEBUG:
        save_gcs_graph_diagram(subgraph.gcs, Path("subgraph_w_contact_t_pusher.svg"))
        visualize_planar_pushing_trajectory_legacy(
            traj, subgraph.slider.geometry, plan_config.pusher_radius
        )


@pytest.mark.parametrize(
    "subgraph",
    [
        {
            "boundary_conds": False,
            "avoid_object": False,
        },
        {
            "boundary_conds": False,
            "avoid_object": True,
        },
    ],
    indirect=["subgraph"],
    ids=[
        "simple",
        "avoid_object",
    ],
)
def test_subgraph_with_contact_modes_band_sparsity(
    subgraph: NonCollisionSubGraph,
    gcs_options: opt.GraphOfConvexSetsOptions,
):
    """
    This unit test tests much of the code that is implemented in the PlanarPushingPlanner.
    """

    subgraph.config.use_band_sparsity = True

    contact_location_start = PolytopeContactLocation(ContactLocation.FACE, 3)
    source_mode = FaceContactMode.create_from_plan_spec(
        contact_location_start, subgraph.config
    )
    source_mode.set_finger_pos(0.5)

    contact_location_end = PolytopeContactLocation(ContactLocation.FACE, 2)
    target_mode = FaceContactMode.create_from_plan_spec(
        contact_location_end, subgraph.config
    )
    target_mode.set_finger_pos(0.5)

    slider_initial_pose = PlanarPose(0.3, 0, 0)
    source_mode.set_slider_initial_pose(slider_initial_pose)
    source_vertex = subgraph.gcs.AddVertex(source_mode.get_convex_set(), "source")
    source_mode.add_cost_to_vertex(source_vertex)
    source = VertexModePair(source_vertex, source_mode)

    slider_final_pose = PlanarPose(0.5, 0.3, 0.4)
    target_mode.set_slider_final_pose(slider_final_pose)
    target_vertex = subgraph.gcs.AddVertex(target_mode.get_convex_set(), "target")
    target_mode.add_cost_to_vertex(target_vertex)
    target = VertexModePair(target_vertex, target_mode)

    subgraph.connect_with_continuity_constraints(
        contact_location_start.idx, source, outgoing=False
    )
    subgraph.connect_with_continuity_constraints(
        contact_location_end.idx, target, incoming=False
    )

    result = subgraph.gcs.SolveShortestPath(source_vertex, target_vertex, gcs_options)
    assert result.is_success()

    pairs = subgraph.get_all_vertex_mode_pairs()
    pairs["source"] = source
    pairs["target"] = target

    traj = PlanarPushingTrajectory.from_result(
        subgraph.config, result, subgraph.gcs, source_vertex, target_vertex, pairs
    )

    assert_initial_and_final_poses(
        traj, slider_initial_pose, None, slider_final_pose, None
    )

    # Make sure we are not leaving the object
    assert np.all(
        [
            np.abs(p_BP) <= 1.0
            for knot_point in traj.path_knot_points
            for p_BP in knot_point.p_BPs  # type: ignore
        ]
    )

    if subgraph.config.avoid_object:
        first_segment = np.hstack(traj.path_knot_points[1].p_BPs)  # type: ignore
        assert_object_is_avoided(
            subgraph.slider.geometry,
            first_segment,
            min_distance=0.001,
            start_idx=2,
            end_idx=-2,
        )
        second_segment = np.hstack(traj.path_knot_points[2].p_BPs)  # type: ignore
        assert_object_is_avoided(
            subgraph.slider.geometry,
            second_segment,
            min_distance=0.001,
            start_idx=2,
            end_idx=-2,
        )

    if DEBUG:
        save_gcs_graph_diagram(subgraph.gcs, Path("subgraph_w_contact.svg"))
        visualize_planar_pushing_trajectory(
            traj, visualize_knot_points=True, save=True, filename="debug_file"
        )
