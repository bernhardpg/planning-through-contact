# NOTE: All of this will be refactored in the not too distant future.

import numpy as np
from deprecated.geometry.collision_pair import CollisionPair
from deprecated.geometry.contact_mode import ContactModeType, PositionModeType
from deprecated.geometry.rigid_body import RigidBody
from deprecated.planning.gcs import GcsContactPlanner
from deprecated.planning.graph_builder import ContactModeConfig
from deprecated.visualize.visualize import animate_positions, plot_positions_and_forces
from geometry.bezier import BezierCurve
from pydrake.math import eq

# TODO remove
# flake8: noqa

# TODO add a guard that makes sure all bodies in all pairs have same dimension?

# TODO:
# Things to clean up:
# - [ ] External forces
# - [ ] Weights for costs
# - [X] Unactuated bodies
# - [ ] GCSContactPlanner should be removed and replaced
# - [X] Rigid bodies collection
# - [X] Position variables, decision variables, force variables

# - [ ] Position modes
# - [ ] Specifying some mode constraints for source and target config (wait with this until I have fixed position modes too)
# - [ ] Automatic collision_pair generation (wait with this until I have fixed position modes)


def plan_for_box_pushing():
    # Bezier curve params
    problem_dim = 2
    bezier_curve_order = 2

    mass = 1  # kg
    g = 9.81  # m/s^2
    mg = mass * g
    box_width = 2
    box_height = 1
    friction_coeff = 0.5

    finger = RigidBody(
        dim=problem_dim,
        position_curve_order=bezier_curve_order,
        name="f",
        geometry="point",
        actuated=True,
    )
    box = RigidBody(
        dim=problem_dim,
        position_curve_order=bezier_curve_order,
        name="b",
        geometry="box",
        width=box_width,
        height=box_height,
        actuated=False,
    )
    ground = RigidBody(
        dim=problem_dim,
        position_curve_order=bezier_curve_order,
        name="g",
        geometry="box",
        width=20,
        height=box_height,
        actuated=True,
    )
    rigid_bodies = [finger, box, ground]

    x_f = finger.pos_x
    y_f = finger.pos_y
    x_b = box.pos_x
    y_b = box.pos_y
    x_g = ground.pos_x
    y_g = ground.pos_y

    p1 = CollisionPair(
        finger,
        box,
        friction_coeff,
        position_mode=PositionModeType.LEFT,
    )
    p2 = CollisionPair(
        box,
        ground,
        friction_coeff,
        position_mode=PositionModeType.TOP,
    )
    collision_pairs = [p1, p2]

    # Specify problem
    no_ground_motion = [eq(x_g, 0), eq(y_g, -1)]
    additional_constraints = [
        *no_ground_motion,
    ]
    source_config = ContactModeConfig(
        modes={
            p1.name: ContactModeType.NO_CONTACT,
            p2.name: ContactModeType.ROLLING,
        },
        additional_constraints=[
            eq(x_f, 0),
            eq(y_f, 0.6),
            eq(x_b, 6.0),
            eq(y_b, box_height),
        ],
    )
    target_config = ContactModeConfig(
        modes={
            p1.name: ContactModeType.NO_CONTACT,
            p2.name: ContactModeType.ROLLING,
        },
        additional_constraints=[eq(x_b, 10.0), eq(y_b, box_height), eq(x_f, 0.0)],
    )

    # TODO this is very hardcoded
    gravitational_jacobian = np.array([[0, -1, 0, -1, 0, -1]]).T
    external_forces = gravitational_jacobian.dot(mg)

    planner = GcsContactPlanner(
        rigid_bodies,
        collision_pairs,
        external_forces,
        additional_constraints,
        allow_sliding=True,
    )
    planner.add_source_config(source_config)
    planner.add_target_config(target_config)
    planner.build_graph(prune=False)
    planner.save_graph_diagram("graph_box_pushing.svg")
    planner.allow_revisits_to_vertices(1)
    planner.save_graph_diagram("graph_box_pushing_with_revisits.svg")

    # TODO add weights here
    planner.add_position_continuity_constraints()
    planner.add_position_path_length_cost()
    planner.add_force_path_length_cost()
    planner.add_num_visited_vertices_cost(100)
    planner.add_force_strength_cost()

    result = planner.solve()
    ctrl_points = planner.get_ctrl_points(result)
    (
        pos_curves,
        normal_force_curves,
        friction_force_curves,
    ) = planner.get_curves_from_ctrl_points(ctrl_points)

    plot_positions_and_forces(pos_curves, normal_force_curves, friction_force_curves)
    animate_positions(pos_curves, rigid_bodies)
    return


def plan_for_box_pickup():
    # Bezier curve params
    problem_dim = 2
    bezier_curve_order = 2

    mass = 1  # kg
    g = 9.81  # m/s^2
    mg = mass * g
    box_width = 2
    box_height = 1
    friction_coeff = 0.5

    finger_1 = RigidBody(
        dim=problem_dim,
        position_curve_order=bezier_curve_order,
        name="f1",
        geometry="point",
        actuated=True,
    )
    finger_2 = RigidBody(
        dim=problem_dim,
        position_curve_order=bezier_curve_order,
        name="f2",
        geometry="point",
        actuated=True,
    )
    box = RigidBody(
        dim=problem_dim,
        position_curve_order=bezier_curve_order,
        name="b",
        geometry="box",
        width=box_width,
        height=box_height,
        actuated=False,
    )
    ground = RigidBody(
        dim=problem_dim,
        position_curve_order=bezier_curve_order,
        name="g",
        geometry="box",
        width=20,
        height=box_height,
        actuated=True,
    )
    rigid_bodies = [finger_1, finger_2, box, ground]

    x_f_1 = finger_1.pos_x
    y_f_1 = finger_1.pos_y
    x_f_2 = finger_2.pos_x
    y_f_2 = finger_2.pos_y
    x_b = box.pos_x
    y_b = box.pos_y
    x_g = ground.pos_x
    y_g = ground.pos_y

    # TODO these collision pairs will soon be generated automatically
    p1 = CollisionPair(
        finger_1,
        box,
        friction_coeff,
        position_mode=PositionModeType.LEFT,
    )
    p2 = CollisionPair(
        finger_2,
        box,
        friction_coeff,
        position_mode=PositionModeType.RIGHT,
    )
    p3 = CollisionPair(
        box,
        ground,
        friction_coeff,
        position_mode=PositionModeType.TOP,
    )
    p4 = CollisionPair(
        finger_1,
        ground,
        friction_coeff,
        position_mode=PositionModeType.TOP,
    )
    p5 = CollisionPair(
        finger_2,
        ground,
        friction_coeff,
        position_mode=PositionModeType.TOP,
    )
    collision_pairs = [p1, p2, p3, p4, p5]

    # Specify problem
    no_ground_motion = [eq(x_g, 0), eq(y_g, -1)]
    additional_constraints = [
        *no_ground_motion,
    ]
    source_config = ContactModeConfig(
        modes={
            p1.name: ContactModeType.NO_CONTACT,
            p2.name: ContactModeType.NO_CONTACT,
            p3.name: ContactModeType.ROLLING,
            p4.name: ContactModeType.NO_CONTACT,
            p5.name: ContactModeType.NO_CONTACT,
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
    # TODO make it such that all non-specified modes are automatically not in contact
    target_config = ContactModeConfig(
        modes={
            p1.name: ContactModeType.ROLLING,
            p2.name: ContactModeType.ROLLING,
            p3.name: ContactModeType.NO_CONTACT,
            p4.name: ContactModeType.NO_CONTACT,
            p5.name: ContactModeType.NO_CONTACT,
        },
        additional_constraints=[eq(x_b, 10.0), eq(y_b, 4.0)],
    )

    # TODO this is very hardcoded
    gravitational_jacobian = np.array([[0, -1, 0, -1, 0, -1, 0, -1]]).T
    external_forces = gravitational_jacobian.dot(mg)

    planner = GcsContactPlanner(
        rigid_bodies,
        collision_pairs,
        external_forces,
        additional_constraints,
        allow_sliding=False,
    )
    planner.add_source_config(source_config)
    planner.add_target_config(target_config)
    planner.build_graph(prune=True)
    planner.save_graph_diagram("box_pickup.svg")
    planner.allow_revisits_to_vertices(0)
    planner.save_graph_diagram("box_pickup_w_revists.svg")

    # TODO add weights here
    planner.add_position_continuity_constraints()
    planner.add_position_path_length_cost()
    planner.add_force_path_length_cost()
    planner.add_num_visited_vertices_cost(100)
    planner.add_force_strength_cost()

    result = planner.solve()
    ctrl_points = planner.get_ctrl_points(result)
    (
        pos_curves,
        normal_force_curves,
        friction_force_curves,
    ) = planner.get_curves_from_ctrl_points(ctrl_points)

    plot_positions_and_forces(pos_curves, normal_force_curves, friction_force_curves)
    animate_positions(pos_curves, rigid_bodies)
    return
