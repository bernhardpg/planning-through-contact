import numpy as np
from pydrake.math import eq, le

from geometry.bezier import BezierCurve
from geometry.contact import CollisionPair, RigidBody
from planning.gcs import GcsContactPlanner
from visualize.visualize import animate_positions, plot_positions_and_forces

# TODO
# Plan:
# - Generalize visualization for arbitraty number of forces

# For forces I need:
# - start position -> ctrl point for positions: of the contact location
# - unit vector -> Unit vector for that collision pair
# - force -> ctrl points for forces


# - Deal with another rigid body in the scene:


def plan_for_one_d_pusher_2():
    # Bezier curve params
    dim = 2
    order = 2

    mass = 1  # kg
    g = 9.81  # m/s^2
    mg = mass * g
    box_width = 2
    box_height = 1
    friction_coeff = 0.5

    finger = RigidBody(dim=dim, order=order, name="finger", point_contact=True)
    box_1 = RigidBody(dim=dim, order=order, name="box_1")
    box_2 = RigidBody(dim=dim, order=order, name="box_2")
    ground = RigidBody(dim=dim, order=order, name="ground", point_contact=True)

    x_f = finger.pos.x[0, :]
    y_f = finger.pos.x[1, :]
    x_b_1 = box_1.pos.x[0, :]
    y_b_1 = box_1.pos.x[1, :]
    x_g = ground.pos.x[0, :]
    y_g = ground.pos.x[1, :]
    x_b_2 = box_2.pos.x[0, :]
    y_b_2 = box_2.pos.x[1, :]

    box_1.register_collision_geometry("left_edge", x_b_1 - box_width)
    box_1.register_collision_geometry("right_edge", x_b_1 + box_width)
    box_1.register_collision_geometry("bottom_edge", y_b_1 - box_height)

    box_2.register_collision_geometry("left_edge", x_b_2 - box_width)
    box_2.register_collision_geometry("bottom_edge", y_b_2 - box_height)

    pair_finger_box_1 = CollisionPair(
        finger,
        "com_x",
        box_1,
        "left_edge",
        friction_coeff,
        n_hat=np.array([[1], [0]]),
    )
    pair_ground_box_1 = CollisionPair(
        ground,
        "com_y",
        box_1,
        "bottom_edge",
        friction_coeff,
        n_hat=np.array([[0], [1]]),
    )
    pair_ground_box_2 = CollisionPair(
        ground,
        "com_y",
        box_2,
        "bottom_edge",
        friction_coeff,
        n_hat=np.array([[0], [1]]),
    )

    all_pairs = [pair_finger_box_1, pair_ground_box_1, pair_ground_box_2]

    # TODO this is very hardcoded
    gravitational_jacobian = np.array([[0, -1, 0, -1, 0, -1, 0, -1]]).T
    external_forces = gravitational_jacobian.dot(mg)

    unactuated_bodies = ["box_1", "box_2"]

    no_ground_motion = [eq(x_g, 0), eq(y_g, 0)]
    finger_pos_below_box_height = le(y_f, y_b_1 + box_height)
    additional_constraints = [
        *no_ground_motion,
        finger_pos_below_box_height,
        eq(pair_ground_box_1.lam_n, mg),
    ]

    source_constraints = [
        eq(x_f, 0),
        eq(y_f, 0.6),
        eq(x_b_1, 4.0),
        eq(y_b_1, box_height),
        eq(x_b_2, 14.0),
        eq(y_b_2, box_height),
    ]
    target_constraints = [eq(x_f, 0.0), eq(x_b_1, 5)]

    planner = GcsContactPlanner(
        all_pairs,
        additional_constraints,
        external_forces,
        unactuated_bodies,
    )

    planner.add_source(source_constraints)
    planner.add_target(target_constraints)
    planner.allow_revisits_to_vertices(1)  # TODO not optimal to call this here

    # TODO add weights here
    planner.add_position_continuity_constraints()
    planner.add_position_path_length_cost()
    planner.add_num_visited_vertices_cost(100)

    planner.save_graph_diagram("graph.svg")
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
    animate_positions(pos_curves, box_width=box_width, box_height=box_height)
    return
