import numpy as np
from pydrake.math import eq, le

from geometry.bezier import BezierCurve
from geometry.contact import CollisionPair, RigidBody
from planning.gcs import GcsContactPlanner
from visualize.visualize import animate_positions

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

    # Physical params
    mass = 1  # kg
    g = 9.81  # m/s^2
    mg = mass * g
    box_width = 2
    box_height = 1
    friction_coeff = 0.5

    # Define variables
    finger = RigidBody(dim=dim, order=order, name="finger", point_contact=True)
    box = RigidBody(dim=dim, order=order, name="box")
    ground = RigidBody(dim=dim, order=order, name="ground", point_contact=True)

    x_f = finger.pos.x[0, :]
    y_f = finger.pos.x[1, :]
    x_b = box.pos.x[0, :]
    y_b = box.pos.x[1, :]
    x_g = ground.pos.x[0, :]
    y_g = ground.pos.x[1, :]

    left_edge = x_b - box_width
    bottom_edge = y_b - box_height

    box.register_collision_geometry("left_edge", left_edge)
    box.register_collision_geometry("bottom_edge", bottom_edge)

    pair_finger_box = CollisionPair(
        finger,
        "com_x",
        box,
        "left_edge",
        friction_coeff,
        n_hat=np.array([[1], [0]]),
    )
    pair_ground_box = CollisionPair(
        ground,
        "com_y",
        box,
        "bottom_edge",
        friction_coeff,
        n_hat=np.array([[0], [1]]),
    )

    all_pairs = [pair_finger_box, pair_ground_box]

    # TODO this is very hardcoded
    gravitational_jacobian = np.array([[0, -1, 0, -1, 0, -1]]).T
    external_forces = gravitational_jacobian.dot(mg)

    unactuated_bodies = ["box"]

    no_ground_motion = [eq(x_g, 0), eq(y_g, 0)]
    no_box_y_motion = eq(y_b, box_height)
    finger_pos_below_box_height = le(y_f, y_b + box_height)
    additional_constraints = [
        *no_ground_motion,
        no_box_y_motion,
        finger_pos_below_box_height,
        eq(pair_ground_box.lam_n, mg),
    ]

    source_constraints = [eq(x_f, 0), eq(y_f, 0.6), eq(x_b, 4.0)]
    target_constraints = [eq(x_f, 0.0), eq(x_b, 15.0)]

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

    animate_positions(pos_curves, box_width=box_width, box_height=box_height)
    # Plot
    #    plt.plot(np.hstack(list(curves.values())))
    #    plt.legend(list(curves.keys()))
    return
