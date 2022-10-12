import numpy as np
from pydrake.math import eq, le

from geometry.bezier import BezierCurve
from geometry.contact import CollisionPair, RigidBody
from planning.gcs import GcsContactPlanner
from visualize.visualize import animate_positions

# TODO Plan:
# DONE:
# 1. Automatically enumerate contact mode combinations from hand-specified modes
# 2. Make an object for handling this
# 3. Extend to y-axis
# 4. Automatically create mode constraints
# 5. Jacobians, normal_vec, friction_vec

# Plan going forward:
# 6. Code cleanup
# 7. Functionality for adding source and target constraints in a nice way
# 8. Deal with multiple visits to the same node
# 9. Two fingers picking up box
# 10. Fix energy cost, should be easy!
# 11. For 3D: extend with friction rays at some points


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
    finger = RigidBody(dim=dim, order=order, name="finger")
    box = RigidBody(dim=dim, order=order, name="box")
    ground = RigidBody(dim=dim, order=order, name="ground")

    x_f = finger.pos.x[0, :]
    y_f = finger.pos.x[1, :]
    x_b = box.pos.x[0, :]
    y_b = box.pos.x[1, :]
    x_g = ground.pos.x[0, :]
    y_g = ground.pos.x[1, :]

    # NOTE this is the stuff the jacobians will replace
    sdf_finger_box = x_b - x_f - box_width
    sdf_ground_box = y_b - y_g - box_height

    pair_finger_box = CollisionPair(
        finger,
        box,
        friction_coeff,
        sdf_finger_box,
        n_hat=np.array([[1], [0]]),
    )
    pair_ground_box = CollisionPair(
        ground,
        box,
        friction_coeff,
        sdf_ground_box,
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
