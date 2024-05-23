import numpy as np
import pydrake.symbolic as sym
import pytest

from planning_through_contact.geometry.collision_geometry.box_2d import Box2d
from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    ContactLocation,
    PolytopeContactLocation,
)
from planning_through_contact.geometry.in_plane.contact_force import (
    ContactForce,
    ContactForceDefinition,
)
from planning_through_contact.geometry.rigid_body import RigidBody
from planning_through_contact.tools.types import NpExpressionArray
from tests.utils import assert_np_expression_array_eq


@pytest.fixture
def box_fric_coeff_lam():
    mu = 0.5
    box = RigidBody("box", Box2d(0.15, 0.15), mass=0.1)
    lam = sym.Variable("lam")
    return box, mu, lam


def test_contact_force_face_rolling(box_fric_coeff_lam):
    box, mu, lam = box_fric_coeff_lam

    loc = PolytopeContactLocation(ContactLocation.FACE, 0)
    p_Bc: NpExpressionArray = box.geometry.get_p_Bc_from_lam(lam, loc)  # type: ignore

    force_def = ContactForceDefinition("test_force", box, mu, loc)

    force = ContactForce.from_definition(force_def, p_Bc)

    assert force.location == ContactLocation.FACE
    assert len(force.variables) == 2

    assert force.normal_vec is not None
    n, t = box.geometry.get_norm_and_tang_vecs_from_location(loc)
    assert np.allclose(force.normal_vec, n)

    assert force.force.shape == (2, 1)
    assert force.pos.shape == (2, 1)

    expected_force = force.variables[0] * n + force.variables[1] * t
    assert_np_expression_array_eq(force.force, expected_force)


def test_contact_force_face_sticking(box_fric_coeff_lam):
    box, mu, lam = box_fric_coeff_lam
    loc = PolytopeContactLocation(ContactLocation.FACE, 0)
    p_Bc: NpExpressionArray = box.geometry.get_p_Bc_from_lam(lam, loc)  # type: ignore

    force_def = ContactForceDefinition(
        "test_force", box, mu, loc, fixed_to_friction_cone_boundary="LEFT"
    )

    force = ContactForce.from_definition(force_def, p_Bc)

    assert force.location == ContactLocation.FACE
    assert (
        len(force.variables) == 1
    )  # only one variable when fixed to friction cone side

    assert force.normal_vec is not None
    n, t = box.geometry.get_norm_and_tang_vecs_from_location(loc)
    assert np.allclose(force.normal_vec, n)

    assert force.force.shape == (2, 1)
    assert force.pos.shape == (2, 1)

    fric = -mu * force.variables[0]
    expected_force = force.variables[0] * n + fric * t
    assert_np_expression_array_eq(force.force, expected_force)


def test_contact_force_vertex(box_fric_coeff_lam):
    box, mu, _ = box_fric_coeff_lam
    loc = PolytopeContactLocation(ContactLocation.VERTEX, 0)
    p_Bc = box.geometry.vertices[loc.idx]

    force_def = ContactForceDefinition("test_force", box, mu, loc)

    force = ContactForce.from_definition(force_def, p_Bc)

    assert force.location == ContactLocation.VERTEX
    assert np.allclose(force.pos, p_Bc)
    assert len(force.variables) == 2
    assert force.normal_vec is None
    assert force.force.shape == (2, 1)
    assert force.pos.shape == (2, 1)

    for v in force.force.flatten():
        assert isinstance(
            v, sym.Variable
        )  # we should have pure variables on the vertex contacts
