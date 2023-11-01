from planning_through_contact.geometry.collision_geometry.box_2d import Box2d
from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    ContactLocation,
    ContactMode,
    PolytopeContactLocation,
)
from planning_through_contact.geometry.in_plane.contact_pair import (
    ContactPairDefinition,
)
from planning_through_contact.geometry.in_plane.contact_scene import (
    ContactScene,
    ContactSceneDefinition,
)
from planning_through_contact.geometry.rigid_body import RigidBody
from planning_through_contact.planning.contact_mode_motion_planner import (
    ContactModeMotionPlanner,
)
from tests.utils import (
    assert_num_vars_in_expression_array,
    assert_num_vars_in_formula_array,
)


def test_contact_scene():
    box = RigidBody("box", Box2d(0.15, 0.15), mass=0.1)
    loc_box_robot = PolytopeContactLocation(ContactLocation.FACE, 3)
    robot = RigidBody("robot", Box2d(0.05, 0.03), mass=0.03, is_actuated=True)
    loc_robot = PolytopeContactLocation(ContactLocation.FACE, 2)

    table = RigidBody("table", Box2d(0.5, 0.03), mass=0.03, is_actuated=True)
    loc_table = PolytopeContactLocation(ContactLocation.FACE, 0)
    loc_box_table = PolytopeContactLocation(ContactLocation.VERTEX, 2)

    face_contact = ContactPairDefinition(
        "contact_1", box, loc_box_robot, robot, loc_robot
    )
    point_contact = ContactPairDefinition(
        "contact_2", box, loc_box_table, table, loc_table
    )

    scene_def = ContactSceneDefinition(
        [table, box, robot],
        [point_contact, face_contact],
        table,
    )
    contact_modes = {
        "contact_1": ContactMode.ROLLING,
        "contact_2": ContactMode.ROLLING,
    }
    scene = scene_def.create_scene(contact_modes)

    assert len(scene.rigid_bodies) == 3

    # face contact: c_n, c_f * 2, lam = 5 vars
    # point contact: cos, sin, f_x, f_y, c_n, c_f, lam, pos in both frames = 11
    assert len(scene.variables) == 16

    assert len(scene.contact_pairs) == 2

    assert scene.unactuated_bodies == [box]

    forces = scene._get_contact_forces_acting_on_body(box)
    # 1 force from table point contact, 2 forces from line contact with box
    assert len(forces) == 3

    for f in forces:
        assert_num_vars_in_expression_array(
            f, 1
        )  # each row should only contain a single variable

    R_WB = scene._get_rotation_to_W(box)
    assert R_WB.shape == (2, 2)

    assert_num_vars_in_expression_array(R_WB, 1)  # cos or sin in each entry

    (
        force_balance,
        torque_balance,
        _,
    ) = scene.create_static_equilibrium_constraints_for_body(box)

    assert force_balance.shape == (2, 1)

    # Each force_balance term should contain 3 force terms (i.e. f_x, c_n_left, c_n_right and cos for rotation of gravity force)
    # This will change if the geometry is not a box where the sides are axis-aligned!
    assert_num_vars_in_formula_array(force_balance, 4)

    contact_points = scene._get_contact_points_on_body(box)
    assert len(contact_points) == 3  # one for table, two for robot

    assert contact_points[0].dtype == float  # constant contact with table
    assert contact_points[1].dtype == object
    assert contact_points[2].dtype == object

    assert_num_vars_in_expression_array(contact_points[1], 1)  # only lam
    assert_num_vars_in_expression_array(contact_points[2], 1)  # only lam

    assert torque_balance.shape == (
        1,
    )  # TODO(bernhardpg): This can just be an expression

    # 6 forces, 1 contact point var (lam)
    assert_num_vars_in_formula_array(torque_balance, 7)


# TODO(bernhardpg): Write unit tests for the ContactSceneCrtrlPoint
# TODO(bernhardpg): Write unit tests for ContaactModeMotionPlanner (needs a new name)
