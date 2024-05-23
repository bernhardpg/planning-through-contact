import numpy as np
import pydrake.symbolic as sym

from planning_through_contact.geometry.collision_geometry.box_2d import Box2d
from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    ContactLocation,
    ContactMode,
    PolytopeContactLocation,
)
from planning_through_contact.geometry.in_plane.contact_pair import (
    ContactFrameConstraints,
    ContactPairDefinition,
    FaceOnFaceContact,
    LineContactConstraints,
    PointContactConstraints,
    PointOnFaceContact,
)
from planning_through_contact.geometry.in_plane.contact_point import ContactPoint
from planning_through_contact.geometry.rigid_body import RigidBody
from planning_through_contact.tools.utils import convert_formula_to_lhs_expression
from tests.utils import assert_formula_array_degree, assert_num_vars_in_formula_array


def test_contact_pair_face_on_face():
    box = RigidBody("box", Box2d(0.15, 0.15), mass=0.1)
    loc_box = PolytopeContactLocation(ContactLocation.FACE, 3)
    robot = RigidBody("robot", Box2d(0.05, 0.03), mass=0.03)
    loc_robot = PolytopeContactLocation(ContactLocation.FACE, 1)

    pair_def = ContactPairDefinition("pair", box, loc_box, robot, loc_robot)

    pair = pair_def.create_pair(ContactMode.ROLLING)

    assert isinstance(pair, FaceOnFaceContact)

    assert pair.body_A == box
    assert pair.body_A_contact_location == loc_box
    assert pair.body_B == robot
    assert pair.body_B_contact_location == loc_robot

    assert pair.shortest_face_length == 0.03
    assert pair.longest_face_length == 0.15

    # There should be one contact force for each side of the face
    assert len(pair.contact_point_A.contact_forces) == 2

    # Should only be one contact point with face-on-face contact
    assert len(pair.contact_points) == 1

    # There should be no relative rotation between two boxes with face-on-face contact
    assert np.allclose(pair.R_AB, np.eye(2)) or np.allclose(pair.R_AB, -np.eye(2))

    assert pair.get_nonfixed_contact_point() == pair.contact_point_A

    # c_n, c_f, c_n, c_f, lam
    assert len(pair.variables) == 5

    cs = pair.create_convex_hull_bounds()
    assert cs.shape == (2,)
    # The only variables in the contact pos constraints should be lambda
    lam = pair.variables[-1]
    for c in cs:
        expr = convert_formula_to_lhs_expression(c)
        vars = expr.GetVariables()
        assert len(vars) == 1
        var = list(vars)[0]
        assert var.EqualTo(lam)

    # 3 friction cone constraints per force
    assert pair.create_friction_cone_constraints().shape == (2 * 3, 1)

    cs = pair.create_constraints()
    assert isinstance(cs, LineContactConstraints)

    assert cs.friction_cone.shape == (2 * 3, 1)
    assert cs.convex_hull_bounds.shape == (2,)

    sq_contact_forces = pair.create_squared_contact_forces()
    assert isinstance(sq_contact_forces, sym.Expression)
    expected_vars = sym.Variables(pair.variables[:4])
    assert sq_contact_forces.GetVariables().EqualTo(expected_vars)

    sq_for_box = pair.get_squared_contact_forces_for_body(box)
    assert isinstance(sq_contact_forces, sym.Expression)
    assert sq_for_box.GetVariables().EqualTo(expected_vars)


def test_contact_pair_point_on_face():
    box = RigidBody("box", Box2d(0.15, 0.15), mass=0.1)
    loc_box = PolytopeContactLocation(ContactLocation.FACE, 3)
    robot = RigidBody("robot", Box2d(0.05, 0.03), mass=0.03)
    loc_robot = PolytopeContactLocation(ContactLocation.VERTEX, 2)

    pair_def = ContactPairDefinition("pair", box, loc_box, robot, loc_robot)

    pair = pair_def.create_pair(ContactMode.ROLLING)

    assert isinstance(pair, PointOnFaceContact)

    assert pair.body_A == box
    assert pair.body_A_contact_location == loc_box
    assert pair.body_B == robot
    assert pair.body_B_contact_location == loc_robot

    assert pair.R_AB.shape == (2, 2)
    for c in pair.R_AB.flatten():
        assert isinstance(c, sym.Variable) or isinstance(c, sym.Expression)

    assert pair.p_AB_A.shape == (2, 1)
    for c in pair.p_AB_A.flatten():
        assert isinstance(c, sym.Variable)

    assert pair.p_BA_B.shape == (2, 1)
    for c in pair.p_BA_B.flatten():
        assert isinstance(c, sym.Variable)

    # One point in each body frame
    assert len(pair.contact_points) == 2

    assert pair.orientation_variables.shape == (2,)

    nonfixed_cp = pair.get_nonfixed_contact_point()
    assert isinstance(nonfixed_cp, ContactPoint)
    assert (
        nonfixed_cp.name == "pair_box"
    )  # the contact point in the robot should be fixed
    for e in nonfixed_cp.contact_position.flatten():
        assert isinstance(e, sym.Expression)  # should not be constant float values

    # cos, sin, c_n, c_f, lam, f_x, f_y, p_AB_x and y, p_BA_x and y
    assert len(pair.variables) == 11

    eq_point_cs = pair.create_equal_contact_point_constraints()
    assert isinstance(eq_point_cs, ContactFrameConstraints)
    assert eq_point_cs.in_frame_A.shape == (2,)
    assert eq_point_cs.in_frame_B.shape == (2,)

    assert_formula_array_degree(eq_point_cs.in_frame_A, 1)
    assert eq_point_cs.type_A == "linear"

    assert_formula_array_degree(eq_point_cs.in_frame_B, 2)
    assert eq_point_cs.type_B == "quadratic"

    # lam, cos, sin and either p_AB x or y
    assert_num_vars_in_formula_array(eq_point_cs.in_frame_A, 4)
    # lam, cos, sin and either p_BA x or y
    assert_num_vars_in_formula_array(eq_point_cs.in_frame_B, 4)

    eq_rel_pos_cs = pair.create_equal_rel_position_constraints()
    assert isinstance(eq_rel_pos_cs, ContactFrameConstraints)
    assert eq_rel_pos_cs.in_frame_A.shape == (2,)
    assert eq_rel_pos_cs.in_frame_B.shape == (2,)

    assert_formula_array_degree(eq_rel_pos_cs.in_frame_A, 2)
    assert eq_rel_pos_cs.type_A == "quadratic"

    assert_formula_array_degree(eq_rel_pos_cs.in_frame_B, 2)
    assert eq_rel_pos_cs.type_B == "quadratic"

    # each row will contain 5 variables:
    # p_AB x or y, cos, sin, p_BA x and y
    assert_num_vars_in_formula_array(eq_rel_pos_cs.in_frame_A, 5)
    # p_BA x or y, cos, sin, p_AB x and y
    assert_num_vars_in_formula_array(eq_rel_pos_cs.in_frame_B, 5)

    eq_force_cs = pair.create_equal_and_opposite_forces_constraint()
    assert isinstance(eq_force_cs, ContactFrameConstraints)
    assert eq_force_cs.in_frame_A.shape == (2,)
    assert eq_force_cs.in_frame_B.shape == (2,)

    assert_formula_array_degree(eq_force_cs.in_frame_A, 2)
    assert eq_force_cs.type_A == "quadratic"

    assert_formula_array_degree(eq_force_cs.in_frame_B, 2)
    assert eq_force_cs.type_B == "quadratic"

    # each row will contain 5 variables:
    # c_n or c_f, cos, sin, f_x and f_y
    assert_num_vars_in_formula_array(eq_force_cs.in_frame_A, 5)
    # f_x or f_y, cos, sin, c_n and c_f
    assert_num_vars_in_formula_array(eq_force_cs.in_frame_B, 5)

    so2_const = pair.create_so2_constraint()
    assert isinstance(so2_const, sym.Formula)
    # cos and sine
    assert len(convert_formula_to_lhs_expression(so2_const).GetVariables()) == 2

    rot_bounds = pair.create_rotation_bounds()
    assert len(rot_bounds) == 4
    for r in rot_bounds:
        assert isinstance(r, sym.Formula)
        assert len(convert_formula_to_lhs_expression(r).GetVariables()) == 1

    chull_bounds = pair.create_convex_hull_bounds()
    assert len(chull_bounds) == 2
    for c in chull_bounds:
        # lam
        assert len(convert_formula_to_lhs_expression(c).GetVariables()) == 1

    face_point = pair._get_contact_point_of_type(ContactLocation.FACE)
    for e in face_point.contact_position.flatten():
        assert isinstance(e, sym.Expression)
    vertex_point = pair._get_contact_point_of_type(ContactLocation.VERTEX)
    for n in vertex_point.contact_position.flatten():
        assert isinstance(n, float)

    cs = pair.create_constraints()
    assert isinstance(cs, PointContactConstraints)
