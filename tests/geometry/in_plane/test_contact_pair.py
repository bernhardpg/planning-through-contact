import numpy as np

from planning_through_contact.geometry.collision_geometry.box_2d import Box2d
from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    ContactLocation,
    ContactMode,
    PolytopeContactLocation,
)
from planning_through_contact.geometry.in_plane.contact_pair import (
    ContactPairDefinition,
    FaceOnFaceContact,
)
from planning_through_contact.geometry.rigid_body import RigidBody


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
