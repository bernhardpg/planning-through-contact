import argparse
from typing import Optional

import numpy as np

from geometry.two_d.box_2d import Box2d
from geometry.two_d.contact.contact_pair_2d import ContactPairDefinition
from geometry.two_d.contact.contact_scene_2d import ContactScene2d
from geometry.two_d.contact.types import ContactLocation, ContactMode
from geometry.two_d.equilateral_polytope_2d import EquilateralPolytope2d
from geometry.two_d.rigid_body_2d import PolytopeContactLocation
from planning.contact_mode_motion_planner import ContactModeMotionPlanner
from tools.utils import evaluate_np_expressions_array
from visualize.visualizer_2d import (
    VisualizationCone2d,
    VisualizationForce2d,
    VisualizationPoint2d,
    VisualizationPolygon2d,
    Visualizer2d,
)


def plan_polytope_flipup(
    num_vertices: int,
    contact_vertex: int,
    th_initial: Optional[float],
    th_target: Optional[float],
    sliding: bool = False,
) -> None:
    FRICTION_COEFF = 0.7
    TABLE_HEIGHT = 0.5
    TABLE_WIDTH = 2

    FINGER_HEIGHT = 0.1
    FINGER_WIDTH = 0.2

    POLYTOPE_MASS = 1

    # TODO:
    # 1. Make sure that contact locations for both forces always stay within the polytope
    # 2. contact force displacements should be calculated automatically

    polytope = EquilateralPolytope2d(
        actuated=False,
        name="polytope",
        mass=POLYTOPE_MASS,
        vertex_distance=0.2,
        num_vertices=num_vertices,
    )
    table = Box2d(
        actuated=True,
        name="table",
        mass=None,
        width=TABLE_WIDTH,
        height=TABLE_HEIGHT,
    )
    finger = Box2d(
        actuated=True,
        name="finger",
        mass=None,
        width=FINGER_WIDTH,
        height=FINGER_HEIGHT,
    )
    table_polytope = ContactPairDefinition(
        "contact_1",
        table,
        PolytopeContactLocation(ContactLocation.FACE, 1),
        polytope,
        PolytopeContactLocation(ContactLocation.VERTEX, contact_vertex),
        FRICTION_COEFF,
    )
    polytope_finger = ContactPairDefinition(
        "contact_2",
        polytope,
        PolytopeContactLocation(ContactLocation.FACE, 0),
        finger,
        PolytopeContactLocation(ContactLocation.FACE, 3),
        FRICTION_COEFF,
    )
    contact_scene = ContactScene2d(
        [table, polytope, finger],
        [table_polytope, polytope_finger],
        table,
    )

    # TODO: this should be cleaned up
    MAX_FORCE = POLYTOPE_MASS * 9.81 * 2.0  # only used for mccorimick constraints
    variable_bounds = {
        "contact_1_polytope_c_n": (0.0, MAX_FORCE),
        "contact_1_polytope_c_f": (
            -FRICTION_COEFF * MAX_FORCE,
            FRICTION_COEFF * MAX_FORCE,
        ),
        "contact_1_table_c_n": (0.0, MAX_FORCE),
        "contact_1_table_c_f": (
            -FRICTION_COEFF * MAX_FORCE,
            FRICTION_COEFF * MAX_FORCE,
        ),
        "contact_1_table_lam": (0.0, 1.0),
        "contact_1_sin_th": (-1, 1),
        "contact_1_cos_th": (-1, 1),
        "contact_2_polytope_lam": (0.0, 1.0),
        "contact_2_polytope_c_n": (0, MAX_FORCE),
        "contact_2_polytope_c_f": (0, MAX_FORCE),
        "contact_2_sin_th": (-1, 1),
        "contact_2_cos_th": (-1, 1),
        "contact_2_finger_c_n": (0.0, MAX_FORCE),
        "contact_2_finger_c_f": (
            -FRICTION_COEFF * MAX_FORCE,
            FRICTION_COEFF * MAX_FORCE,
        ),
    }

    if sliding:
        contact_modes = {
            "contact_1": ContactMode.SLIDING_LEFT,
            "contact_2": ContactMode.ROLLING,
        }
        lam_target = 0.6
    else:
        contact_modes = {
            "contact_1": ContactMode.ROLLING,
            "contact_2": ContactMode.ROLLING,
        }
        lam_target = None

    NUM_CTRL_POINTS = 3
    motion_plan = ContactModeMotionPlanner(
        contact_scene,
        NUM_CTRL_POINTS,
        contact_modes,
        variable_bounds,
        use_mccormick_relaxation=True,
    )
    if th_initial is not None:
        motion_plan.constrain_orientation_at_ctrl_point(
            table_polytope, ctrl_point_idx=0, theta=th_initial
        )
    if th_target is not None:
        motion_plan.constrain_orientation_at_ctrl_point(
            table_polytope, ctrl_point_idx=NUM_CTRL_POINTS - 1, theta=th_target
        )
    motion_plan.constrain_contact_position_at_ctrl_point(
        table_polytope, ctrl_point_idx=0, lam_target=0.4
    )
    if lam_target is not None:
        motion_plan.constrain_contact_position_at_ctrl_point(
            table_polytope, ctrl_point_idx=NUM_CTRL_POINTS - 1, lam_target=lam_target
        )
    motion_plan.solve()

    if True:
        CONTACT_COLOR = "dodgerblue4"
        GRAVITY_COLOR = "blueviolet"
        BOX_COLOR = "aquamarine4"
        TABLE_COLOR = "bisque3"
        FINGER_COLOR = "firebrick3"
        body_colors = [TABLE_COLOR, BOX_COLOR, FINGER_COLOR]

        contact_positions_ctrl_points = [
            evaluate_np_expressions_array(pos, motion_plan.result)
            for pos in motion_plan.contact_positions_in_world_frame
        ]
        contact_forces_ctrl_points = [
            evaluate_np_expressions_array(force, motion_plan.result)
            for force in motion_plan.contact_forces_in_world_frame
        ]

        bodies_com_ctrl_points = [
            motion_plan.result.GetSolution(ctrl_point)
            for ctrl_point in motion_plan.body_positions_in_world_frame
        ]

        bodies_orientation_ctrl_points = [
            [motion_plan.result.GetSolution(R_ctrl_point) for R_ctrl_point in R]
            for R in motion_plan.body_orientations
        ]

        viz_com_points = [
            VisualizationPoint2d.from_ctrl_points(com_ctrl_points, GRAVITY_COLOR)
            for com_ctrl_points in bodies_com_ctrl_points
        ]

        viz_contact_positions = [
            VisualizationPoint2d.from_ctrl_points(pos, CONTACT_COLOR)
            for pos in contact_positions_ctrl_points
        ]
        viz_contact_forces = [
            VisualizationForce2d.from_ctrl_points(pos, force, CONTACT_COLOR)
            for pos, force in zip(
                contact_positions_ctrl_points, contact_forces_ctrl_points
            )
        ]

        box_com_ctrl_points = bodies_com_ctrl_points[1]
        # TODO: should not depend explicitly on box
        viz_gravitional_forces = [
            VisualizationForce2d.from_ctrl_points(
                box_com_ctrl_points,
                evaluate_np_expressions_array(force_ctrl_points, motion_plan.result),
                GRAVITY_COLOR,
            )
            for force_ctrl_points in motion_plan.gravitational_forces_in_world_frame
        ]

        friction_cone_angle = np.arctan(FRICTION_COEFF)
        (
            fc_normals,
            fc_positions,
            fc_orientations,
        ) = motion_plan.contact_point_friction_cones
        viz_friction_cones = [
            VisualizationCone2d.from_ctrl_points(
                evaluate_np_expressions_array(pos, motion_plan.result),
                [
                    motion_plan.result.GetSolution(R_ctrl_point)
                    for R_ctrl_point in orientation
                ],
                normal_vec,
                friction_cone_angle,
            )
            for normal_vec, pos, orientation in zip(
                fc_normals, fc_positions, fc_orientations
            )
        ]
        viz_friction_cones_mirrored = [
            VisualizationCone2d.from_ctrl_points(
                evaluate_np_expressions_array(pos, motion_plan.result),
                [
                    motion_plan.result.GetSolution(R_ctrl_point)
                    for R_ctrl_point in orientation
                ],
                -normal_vec,
                friction_cone_angle,
            )
            for normal_vec, pos, orientation in zip(
                fc_normals, fc_positions, fc_orientations
            )
        ]

        viz_polygons = [
            VisualizationPolygon2d.from_ctrl_points(
                com,
                orientation,
                body,
                color,
            )
            for com, orientation, body, color in zip(
                bodies_com_ctrl_points,
                bodies_orientation_ctrl_points,
                contact_scene.rigid_bodies,
                body_colors,
            )
        ]

        viz = Visualizer2d()
        viz.visualize(
            viz_contact_positions + viz_com_points,
            viz_contact_forces + viz_gravitional_forces,
            viz_polygons + viz_friction_cones + viz_friction_cones_mirrored,  # type: ignore
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sliding", help="Use sliding", action="store_true", default=False
    )
    parser.add_argument(
        "--num_vertices", help="Number of vertices for polytope", type=int, default=3
    )
    parser.add_argument("--th_initial", help="Initial angle", type=float, default=0.0)
    parser.add_argument(
        "--th_target", help="Target angle", type=float, default=np.pi / 4
    )
    args = parser.parse_args()

    num_vertices = args.num_vertices
    th_initial = args.th_initial
    th_target = args.th_target
    sliding = args.sliding

    # Set some feasible conditions for the different polytopes
    if sliding:
        if num_vertices == 3:
            contact_vertex = 2
            th_target = np.pi / 4
        elif num_vertices == 4:
            contact_vertex = 2
            th_initial = -np.pi / 4
            th_target = 0
        elif num_vertices == 5:
            contact_vertex = 3
            th_target = np.pi / 5
        else:
            contact_vertex = 3
    else:
        if num_vertices == 3:
            contact_vertex = 2
        elif num_vertices == 4:
            contact_vertex = 2
            th_target = np.pi / 6
        elif num_vertices == 5:
            contact_vertex = 3
            th_target = np.pi / 6
        else:
            contact_vertex = 3

    plan_polytope_flipup(num_vertices, contact_vertex, th_initial, th_target, sliding)
