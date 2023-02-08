import argparse

import numpy as np

from geometry.two_d.box_2d import Box2d
from geometry.two_d.contact.contact_pair_2d import ContactPair2d
from geometry.two_d.contact.contact_scene_2d import ContactScene2d
from geometry.two_d.contact.types import ContactPosition, ContactType
from geometry.two_d.equilateral_polytope_2d import EquilateralPolytope2d
from geometry.two_d.rigid_body_2d import PolytopeContactLocation
from planning.contact_mode_motion_planner import ContactModeMotionPlanner
from tools.utils import evaluate_np_expressions_array
from visualize.visualizer_2d import (
    VisualizationForce2d,
    VisualizationPoint2d,
    VisualizationPolygon2d,
    Visualizer2d,
)


def plan_polytope_flipup(
    num_vertices: int, contact_vertex: int, th_initial: float, th_target: float
) -> None:
    FRICTION_COEFF = 0.7
    TABLE_HEIGHT = 0.5
    TABLE_WIDTH = 2

    FINGER_HEIGHT = 0.1
    FINGER_WIDTH = 0.1

    POLYTOPE_MASS = 1

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
    table_polytope = ContactPair2d(
        "contact_1",
        table,
        PolytopeContactLocation(ContactPosition.FACE, 1),
        polytope,
        PolytopeContactLocation(ContactPosition.VERTEX, contact_vertex),
        ContactType.POINT_CONTACT,
        FRICTION_COEFF,
    )
    polytope_finger = ContactPair2d(
        "contact_2",
        polytope,
        PolytopeContactLocation(ContactPosition.FACE, 0),
        finger,
        PolytopeContactLocation(ContactPosition.VERTEX, 1),
        ContactType.POINT_CONTACT,
        FRICTION_COEFF,
    )
    contact_scene = ContactScene2d(
        [table, polytope, finger],
        [table_polytope, polytope_finger],
        table,
    )

    # TODO: this should be cleaned up
    MAX_FORCE = POLYTOPE_MASS * 9.81 * 2  # only used for mccorimick constraints
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
        "contact_2_polytope_c_n": (0, MAX_FORCE / 2),
        "contact_2_polytope_c_f": (0, MAX_FORCE / 2),
        "contact_2_sin_th": (-1, 1),
        "contact_2_cos_th": (-1, 1),
        "contact_2_finger_c_n": (0.0, MAX_FORCE / 2),
        "contact_2_finger_c_f": (
            -FRICTION_COEFF * MAX_FORCE,
            FRICTION_COEFF * MAX_FORCE,
        ),
    }

    NUM_CTRL_POINTS = 3
    motion_plan = ContactModeMotionPlanner(
        contact_scene, NUM_CTRL_POINTS, variable_bounds
    )
    motion_plan.constrain_orientation_at_ctrl_point(
        table_polytope, ctrl_point_idx=0, theta=th_initial
    )
    motion_plan.constrain_orientation_at_ctrl_point(
        table_polytope, ctrl_point_idx=NUM_CTRL_POINTS - 1, theta=th_target
    )
    motion_plan.fix_contact_positions()
    motion_plan.solve()

    contact_positions_ctrl_points = [
        evaluate_np_expressions_array(pos, motion_plan.result)
        for pos in motion_plan.contact_positions_in_world_frame
    ]
    contact_forces_ctrl_points = [
        evaluate_np_expressions_array(force, motion_plan.result)
        for force in motion_plan.contact_forces_in_world_frame
    ]

    if True:

        CONTACT_COLOR = "brown1"
        GRAVITY_COLOR = "blueviolet"
        BOX_COLOR = "aquamarine4"
        TABLE_COLOR = "bisque3"
        FINGER_COLOR = "brown3"
        body_colors = [TABLE_COLOR, BOX_COLOR, FINGER_COLOR]

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

        bodies_com_ctrl_points = [
            motion_plan.result.GetSolution(ctrl_point)
            for ctrl_point in motion_plan.body_positions_in_world_frame
        ]

        viz_com_points = [
            VisualizationPoint2d.from_ctrl_points(com_ctrl_points, GRAVITY_COLOR)
            for com_ctrl_points in bodies_com_ctrl_points
        ]

        box_com_ctrl_points = bodies_com_ctrl_points[1]
        # TODO: should not depend explicitly on box
        viz_gravitional_forces = [
            VisualizationForce2d.from_ctrl_points(
                box_com_ctrl_points,
                motion_plan.result.GetSolution(force_ctrl_points),
                GRAVITY_COLOR,
            )
            for force_ctrl_points in motion_plan.gravitational_forces_in_world_frame
        ]

        bodies_orientation_ctrl_points = [
            [motion_plan.result.GetSolution(R_ctrl_point) for R_ctrl_point in R]
            for R in motion_plan.body_orientations
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
            viz_polygons,
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
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

    # Decide contact location based on shape
    if num_vertices == 3:
        contact_vertex = 2
    elif num_vertices == 4:
        contact_vertex = 2
    elif num_vertices == 5:
        contact_vertex = 3
    else:
        contact_vertex = 3

    plan_polytope_flipup(num_vertices, contact_vertex, th_initial, th_target)
