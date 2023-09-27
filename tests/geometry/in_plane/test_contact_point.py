import numpy as np
import pydrake.symbolic as sym
import pytest

from planning_through_contact.geometry.collision_geometry.box_2d import Box2d
from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    ContactLocation,
    PolytopeContactLocation,
)
from planning_through_contact.geometry.in_plane.contact_force import (
    ContactForceDefinition,
)
from planning_through_contact.geometry.in_plane.contact_point import ContactPoint
from planning_through_contact.geometry.rigid_body import RigidBody
from tests.utils import assert_np_expression_array_eq


@pytest.fixture
def mu_box():
    mu = 0.5
    box = RigidBody("box", Box2d(0.15, 0.15), mass=0.1)
    return mu, box


def test_contact_point_face_rolling(mu_box):
    mu, box = mu_box
    loc = PolytopeContactLocation(ContactLocation.FACE, 0)
    force_def = ContactForceDefinition(f"test_force", mu, loc, box.geometry)

    cp = ContactPoint(
        force_def.body_geometry,
        force_def.location,
        [force_def],
        force_def.friction_coeff,
        name=f"test_contact_point",
    )

    assert cp.contact_position.shape == (2, 1)
    for e in cp.contact_position.flatten():
        assert isinstance(e, sym.Expression)

    assert len(cp.contact_forces) == 1
    cf = cp.contact_forces[0]
    assert cf.location == force_def.location.pos
    assert_np_expression_array_eq(cf.pos, cp.contact_position)  # type: ignore

    # c_n, c_f, lam
    assert len(cp.variables) == 3

    assert len(cp.get_contact_positions()) == 1
    assert len(cp.get_contact_forces()) == 1

    fc_consts = cp.create_friction_cone_constraints()
    assert fc_consts.shape == (3, 1)


def test_contact_point_face_sticking(mu_box):
    mu, box = mu_box
    loc = PolytopeContactLocation(ContactLocation.FACE, 0)
    force_def = ContactForceDefinition(
        f"test_force", mu, loc, box.geometry, fixed_to_friction_cone_boundary="RIGHT"
    )

    cp = ContactPoint(
        force_def.body_geometry,
        force_def.location,
        [force_def],
        force_def.friction_coeff,
        name=f"test_contact_point",
    )

    assert cp.contact_position.shape == (2, 1)
    for e in cp.contact_position.flatten():
        assert isinstance(e, sym.Expression)

    assert len(cp.contact_forces) == 1
    cf = cp.contact_forces[0]
    assert cf.location == force_def.location.pos
    assert_np_expression_array_eq(cf.pos, cp.contact_position)  # type: ignore

    # c_n, lam
    assert len(cp.variables) == 2

    assert len(cp.get_contact_positions()) == 1
    assert len(cp.get_contact_forces()) == 1

    fc_consts = cp.create_friction_cone_constraints()
    assert fc_consts.shape == (1, 1)


def test_contact_point_two_forces(mu_box):
    mu, box = mu_box
    loc = PolytopeContactLocation(ContactLocation.FACE, 0)
    f1_def = ContactForceDefinition(
        f"test_force_1", mu, loc, box.geometry, displacement=0.02
    )
    f2_def = ContactForceDefinition(
        f"test_force_2", mu, loc, box.geometry, displacement=-0.02
    )
    cp = ContactPoint(
        f1_def.body_geometry,
        f1_def.location,
        [f1_def, f2_def],
        f1_def.friction_coeff,
        name=f"test_contact_point",
    )

    assert len(cp.contact_forces) == 2

    # (c_n, c_f) x2 + lam
    assert len(cp.variables) == 5

    assert len(cp.get_contact_positions()) == 2
    assert len(cp.get_contact_forces()) == 2

    fc_consts = cp.create_friction_cone_constraints()
    assert fc_consts.shape == (3 * 2, 1)


def test_contact_point_vertex_rolling(mu_box):
    mu, box = mu_box
    loc = PolytopeContactLocation(ContactLocation.VERTEX, 0)
    force_def = ContactForceDefinition(f"test_force", mu, loc, box.geometry)

    cp = ContactPoint(
        force_def.body_geometry,
        force_def.location,
        [force_def],
        force_def.friction_coeff,
        name=f"test_contact_point",
    )

    assert cp.contact_position.shape == (2, 1)
    for e in cp.contact_position.flatten():
        assert isinstance(e, float)

    cf = cp.contact_forces[0]
    assert cf.location == force_def.location.pos
    assert np.allclose(cf.pos, cp.contact_position)

    # c_n, c_f
    assert len(cp.variables) == 2

    assert len(cp.get_contact_positions()) == 1
    assert len(cp.get_contact_forces()) == 1

    # Should not be possible to make friction cone constraints for a vertex contact
    with pytest.raises(ValueError) as res:
        cp.create_friction_cone_constraints()
