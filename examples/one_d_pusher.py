import numpy as np
from pydrake.math import eq, ge, le

from geometry.bezier import BezierCurve
from geometry.contact import CollisionPair, ContactModeType, PositionModeType, RigidBody
from planning.gcs import GcsContactPlanner
from planning.graph_builder import GraphBuilder, ModeConfig
from visualize.visualize import animate_positions, plot_positions_and_forces

# TODO remove
# flake8: noqa


def plan_w_graph_builder():
    # Bezier curve params
    dim = 2
    order = 2

    mass = 1  # kg
    g = 9.81  # m/s^2
    mg = mass * g
    box_width = 2
    box_height = 1
    friction_coeff = 0.5

    finger_1 = RigidBody(
        dim=dim, position_curve_order=order, name="finger_1", geometry="point"
    )
    finger_2 = RigidBody(
        dim=dim, position_curve_order=order, name="finger_2", geometry="point"
    )
    box = RigidBody(
        dim=dim,
        position_curve_order=order,
        name="box",
        geometry="box",
        width=box_width,
        height=box_height,
    )
    ground = RigidBody(
        dim=dim,
        position_curve_order=order,
        name="ground",
        geometry="box",
        width=20,
        height=box_height,
    )

    x_f_1 = finger_1.pos_x
    y_f_1 = finger_1.pos_y
    x_f_2 = finger_2.pos_x
    y_f_2 = finger_2.pos_y
    x_b = box.pos_x
    y_b = box.pos_y
    x_g = ground.pos_x
    y_g = ground.pos_y

    pair_finger_1_box = CollisionPair(
        finger_1,
        box,
        friction_coeff,
        position_mode=PositionModeType.LEFT,
    )
    pair_finger_2_box = CollisionPair(
        finger_2,
        box,
        friction_coeff,
        position_mode=PositionModeType.RIGHT,
    )
    pair_box_ground = CollisionPair(
        box,
        ground,
        friction_coeff,
        position_mode=PositionModeType.TOP,
    )
    #    pair_finger_1_ground = CollisionPair(
    #        finger_1,
    #        ground,
    #        friction_coeff,
    #        allowed_position_modes=[PositionModeType.TOP],
    #    )
    #    pair_finger_2_ground = CollisionPair(
    #        finger_2,
    #        ground,
    #        friction_coeff,
    #        allowed_position_modes=[PositionModeType.TOP],
    #    )

    pairs = [
        pair_finger_1_box,
        pair_finger_2_box,
        pair_box_ground,
        # pair_finger_1_ground,
        # pair_finger_2_ground,
    ]

    rigid_bodies = [finger_1, finger_2, box, ground]

    # TODO this is very hardcoded
    gravitational_jacobian = np.array([[0, -1, 0, -1, 0, -1, 0, -1]]).T
    external_forces = gravitational_jacobian.dot(mg)

    unactuated_bodies = ["box"]

    no_ground_motion = [eq(x_g, 0), eq(y_g, -1)]
    finger_1_pos_below_box_height = le(y_f_1, y_b + box_height)
    finger_1_pos_above_box_bottom = ge(y_f_1, y_b - box_height)
    finger_2_pos_below_box_height = le(y_f_2, y_b + box_height)
    finger_2_pos_above_box_bottom = ge(y_f_2, y_b - box_height)
    additional_constraints = [
        *no_ground_motion,
    ]

    source = ModeConfig(
        modes={
            pair_finger_1_box.name: ContactModeType.NO_CONTACT,
            pair_finger_2_box.name: ContactModeType.NO_CONTACT,
            pair_box_ground.name: ContactModeType.ROLLING,
        },
        additional_constraints=[
            eq(x_f_1, 0),
            eq(y_f_1, 0.6),
            eq(x_f_2, 10.0),
            eq(y_f_2, 0.6),
            eq(x_b, 6.0),
            eq(y_b, box_height),
        ],
    )
    target = ModeConfig(
        modes={
            pair_finger_1_box.name: ContactModeType.ROLLING,
            pair_finger_2_box.name: ContactModeType.ROLLING,
            pair_box_ground.name: ContactModeType.NO_CONTACT,
        },
        additional_constraints=[eq(x_b, 10.0), eq(y_b, 4.0)],
    )

    graph_builder = GraphBuilder(
        pairs,
        unactuated_bodies,
        external_forces,
        additional_constraints,
    )
    graph_builder.add_source(source)
    graph_builder.add_target(target)
    graph_builder.build_graph()

    breakpoint()
    return


def plan_for_two_fingers():
    # Bezier curve params
    dim = 2
    order = 2

    mass = 1  # kg
    g = 9.81  # m/s^2
    mg = mass * g
    box_width = 2
    box_height = 1
    friction_coeff = 0.5

    finger_1 = RigidBody(
        dim=dim, position_curve_order=order, name="finger_1", geometry="point"
    )
    finger_2 = RigidBody(
        dim=dim, position_curve_order=order, name="finger_2", geometry="point"
    )
    box = RigidBody(
        dim=dim,
        position_curve_order=order,
        name="box",
        geometry="box",
        width=box_width,
        height=box_height,
    )
    ground = RigidBody(
        dim=dim,
        position_curve_order=order,
        name="ground",
        geometry="box",
        width=100,
        height=1,
    )

    x_f_1 = finger_1.pos_x
    y_f_1 = finger_1.pos_y
    x_f_2 = finger_2.pos_x
    y_f_2 = finger_2.pos_y
    x_b = box.pos_x
    y_b = box.pos_y
    x_g = ground.pos_x
    y_g = ground.pos_y

    pair_finger_1_box = CollisionPair(
        finger_1,
        box,
        friction_coeff,
        position_mode=PositionModeType.LEFT,
    )
    pair_finger_2_box = CollisionPair(
        finger_2,
        box,
        friction_coeff,
        position_mode=PositionModeType.RIGHT,
    )
    pair_box_ground = CollisionPair(
        box, ground, friction_coeff, position_mode=PositionModeType.TOP
    )

    bodies = [finger_1, finger_2, box, ground]
    all_pairs = [
        pair_finger_1_box,
        pair_finger_2_box,
        pair_box_ground,
    ]

    # TODO this is very hardcoded
    gravitational_jacobian = np.array([[0, -1, 0, -1, 0, -1, 0, -1]]).T
    external_forces = gravitational_jacobian.dot(mg)

    unactuated_bodies = ["box"]

    no_ground_motion = [eq(x_g, 0), eq(y_g, -1)]
    finger_1_pos_below_box_height = le(y_f_1, y_b + box_height)
    finger_1_pos_above_box_bottom = ge(y_f_1, y_b - box_height)
    finger_2_pos_below_box_height = le(y_f_2, y_b + box_height)
    finger_2_pos_above_box_bottom = ge(y_f_2, y_b - box_height)
    additional_constraints = [
        *no_ground_motion,
        finger_1_pos_below_box_height,
        finger_2_pos_below_box_height,
        finger_1_pos_above_box_bottom,
        finger_2_pos_above_box_bottom,
    ]

    source_constraints = [
        eq(x_f_1, 0),
        eq(y_f_1, 0.6),
        eq(x_f_2, 10.0),
        eq(y_f_2, 0.6),
        eq(x_b, 6.0),
        eq(y_b, box_height),
    ]
    target_constraints = [eq(x_b, 10.0), eq(y_b, 4.0)]

    planner = GcsContactPlanner(
        all_pairs,
        additional_constraints,
        external_forces,
        unactuated_bodies,
        allow_sliding=False,  # if set to True the problem will blow up!
    )

    planner.add_source(source_constraints)
    planner.add_target(target_constraints)

    planner.save_graph_diagram("graph_without_revisits.svg")
    print("Saved graph as diagram")
    planner.allow_revisits_to_vertices(1)  # TODO not optimal to call this here

    # TODO add weights here
    planner.add_position_continuity_constraints()
    planner.add_position_path_length_cost()
    planner.add_force_path_length_cost()
    planner.add_num_visited_vertices_cost(100)
    planner.add_force_strength_cost()

    result = planner.solve()
    vertex_values = planner.get_vertex_values(result)

    normal_forces, friction_forces = planner.get_force_ctrl_points(vertex_values)
    positions = {
        body: planner.get_pos_ctrl_points(vertex_values, body)
        for body in planner.all_bodies
    }

    pos_curves = {
        body: np.concatenate(
            [
                BezierCurve.create_from_ctrl_points(c).eval_entire_interval()
                for c in ctrl_points
            ]
        )
        for body, ctrl_points in positions.items()
    }

    normal_force_curves = {
        pair: np.concatenate(
            [
                BezierCurve.create_from_ctrl_points(
                    points.reshape((1, -1))
                ).eval_entire_interval()
                for points in control_points
            ]
        )
        for pair, control_points in normal_forces.items()
    }

    friction_force_curves = {
        pair: np.concatenate(
            [
                BezierCurve.create_from_ctrl_points(
                    points.reshape((1, -1))
                ).eval_entire_interval()
                for points in control_points
            ]
        )
        for pair, control_points in friction_forces.items()
    }

    plot_positions_and_forces(pos_curves, normal_force_curves, friction_force_curves)
    animate_positions(pos_curves, bodies)
    return


def plan_for_one_box_one_finger():
    # Bezier curve params
    dim = 2
    order = 2

    mass = 1  # kg
    g = 9.81  # m/s^2
    mg = mass * g
    box_width = 2
    box_height = 1
    friction_coeff = 0.5

    finger = RigidBody(
        dim=dim, position_curve_order=order, name="finger", geometry="point"
    )
    box = RigidBody(
        dim=dim,
        position_curve_order=order,
        name="box",
        geometry="box",
        width=box_width,
        height=box_height,
    )
    ground = RigidBody(
        dim=dim,
        position_curve_order=order,
        name="ground",
        geometry="box",
        width=100,
        height=1,
    )

    bodies = [finger, box, ground]
    pair_finger_box = CollisionPair(
        finger,
        box,
        friction_coeff,
        position_mode=PositionModeType.LEFT,
    )
    pair_box_ground = CollisionPair(
        box, ground, friction_coeff, position_mode=PositionModeType.TOP
    )
    all_pairs = [pair_finger_box, pair_box_ground]

    # TODO this is very hardcoded
    gravitational_jacobian = np.array([[0, -1, 0, -1, 0, -1]]).T
    external_forces = gravitational_jacobian.dot(mg)

    unactuated_bodies = ["box"]

    x_f = finger.pos_x
    y_f = finger.pos_y
    x_b = box.pos_x
    y_b = box.pos_y
    x_g = ground.pos_x
    y_g = ground.pos_y

    no_ground_motion = [eq(x_g, 0), eq(y_g, -1)]
    additional_constraints = [
        *no_ground_motion,
        eq(pair_box_ground.lam_n, mg),
    ]

    source_constraints = [
        eq(x_f, 0),
        eq(y_f, 0.6),
        eq(x_b, 4.0),
        eq(y_b, box_height),
    ]
    target_constraints = [eq(x_f, 0.0), eq(x_b, 5)]

    planner = GcsContactPlanner(
        all_pairs,
        additional_constraints,
        external_forces,
        unactuated_bodies,
        allow_sliding=True,
    )

    planner.add_source(source_constraints)
    planner.add_target(target_constraints)

    planner.save_graph_diagram("graph_without_revisits.svg")
    print("Saved graph as diagram")
    planner.allow_revisits_to_vertices(1)  # TODO not optimal to call this here

    # TODO add weights here
    planner.add_position_continuity_constraints()
    planner.add_position_path_length_cost()
    planner.add_num_visited_vertices_cost(100)

    result = planner.solve()
    vertex_values = planner.get_vertex_values(result)

    normal_forces, friction_forces = planner.get_force_ctrl_points(vertex_values)
    positions = {
        body: planner.get_pos_ctrl_points(vertex_values, body)
        for body in planner.all_bodies
    }

    pos_curves = {
        body: np.concatenate(
            [
                BezierCurve.create_from_ctrl_points(c).eval_entire_interval()
                for c in ctrl_points
            ]
        )
        for body, ctrl_points in positions.items()
    }

    normal_force_curves = {
        pair: np.concatenate(
            [
                BezierCurve.create_from_ctrl_points(
                    points.reshape((1, -1))
                ).eval_entire_interval()
                for points in control_points
            ]
        )
        for pair, control_points in normal_forces.items()
    }

    friction_force_curves = {
        pair: np.concatenate(
            [
                BezierCurve.create_from_ctrl_points(
                    points.reshape((1, -1))
                ).eval_entire_interval()
                for points in control_points
            ]
        )
        for pair, control_points in friction_forces.items()
    }

    plot_positions_and_forces(pos_curves, normal_force_curves, friction_force_curves)
    animate_positions(pos_curves, bodies)
    return
