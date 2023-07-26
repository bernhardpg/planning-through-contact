from pathlib import Path
from typing import Tuple

import numpy as np
import pydrake.geometry.optimization as opt

from planning_through_contact.geometry.planar.non_collision import NonCollisionMode
from planning_through_contact.geometry.planar.non_collision_subgraph import (
    VertexModePair,
    gcs_add_edge_with_continuity,
)
from planning_through_contact.geometry.planar.trajectory_builder import (
    PlanarTrajectoryBuilder,
)
from planning_through_contact.geometry.rigid_body import RigidBody
from planning_through_contact.tools.gcs_tools import get_gcs_solution_path
from planning_through_contact.visualize.analysis import save_gcs_graph_diagram
from planning_through_contact.visualize.planar import (
    visualize_planar_pushing_trajectory,
)
from tests.geometry.planar.fixtures import (
    box_geometry,
    initial_and_final_non_collision_mode_one_two_knot_points,
    rigid_body_box,
)


def test_add_continuity_constraints_between_modes(
    initial_and_final_non_collision_mode_one_two_knot_points: Tuple[
        NonCollisionMode, NonCollisionMode
    ],
    rigid_body_box: RigidBody,
) -> None:
    source_mode, target_mode = initial_and_final_non_collision_mode_one_two_knot_points

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

    p_c_W_initial = source_mode.slider_pose.pos() + source_mode.p_BF_initial
    assert np.allclose(traj.p_c_W[:, 0:1], p_c_W_initial)

    p_c_W_final = target_mode.slider_pose.pos() + target_mode.p_BF_final
    assert np.allclose(traj.p_c_W[:, -1:], p_c_W_final)

    DEBUG = False
    if DEBUG:
        save_gcs_graph_diagram(gcs, Path("test_continuity.svg"))
        save_gcs_graph_diagram(gcs, Path("test_continuity_result.svg"), result)
        visualize_planar_pushing_trajectory(traj, rigid_body_box.geometry)
