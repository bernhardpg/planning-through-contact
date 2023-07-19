# import pytest
from pydrake.solvers import MathematicalProgram
from pydrake.symbolic import Expression

from geometry.collision_geometry.box_2d import Box2d
from geometry.collision_geometry.collision_geometry import (
    ContactLocation,
    PolytopeContactLocation,
)
from geometry.planar.planar_contact_modes import FaceContactVariables


def test_face_contact_variables() -> None:
    prog = MathematicalProgram()
    box_geometry = Box2d(width=0.2, height=0.1)
    contact_location = PolytopeContactLocation(ContactLocation.FACE, 0)

    num_knot_points = 4
    time_in_contact = 2

    vars = FaceContactVariables.from_prog(
        prog,
        box_geometry,
        contact_location,
        num_knot_points,
        time_in_contact,
    )

    assert vars.lams.shape == (num_knot_points,)
    assert vars.normal_forces.shape == (num_knot_points,)
    assert vars.friction_forces.shape == (num_knot_points,)
    assert vars.cos_ths.shape == (num_knot_points,)
    assert vars.sin_ths.shape == (num_knot_points,)
    assert vars.p_WB_xs.shape == (num_knot_points,)
    assert vars.p_WB_ys.shape == (num_knot_points,)

    assert all(vars.pv1 == box_geometry.vertices[0])
    assert all(vars.pv2 == box_geometry.vertices[1])

    assert len(vars.R_WBs) == num_knot_points
    for R in vars.R_WBs:
        assert R.shape == (2, 2)

    assert len(vars.p_WBs) == num_knot_points
    for p in vars.p_WBs:
        assert p.shape == (2, 1)

    assert len(vars.f_c_Bs) == num_knot_points
    for f in vars.f_c_Bs:
        assert f.shape == (2, 1)

    assert len(vars.p_c_Bs) == num_knot_points
    for p_c in vars.p_c_Bs:
        assert p_c.shape == (2, 1)

    assert len(vars.v_WBs) == num_knot_points - 1
    for v in vars.v_WBs:
        assert v.shape == (2, 1)

    assert len(vars.cos_th_dots) == num_knot_points - 1
    for c in vars.cos_th_dots:
        assert isinstance(c, Expression)

    assert len(vars.sin_th_dots) == num_knot_points - 1
    for s in vars.cos_th_dots:
        assert isinstance(s, Expression)

    assert len(vars.v_c_Bs) == num_knot_points - 1
    for v in vars.v_c_Bs:
        assert v.shape == (2, 1)

    assert len(vars.omega_WBs) == num_knot_points - 1
    for o in vars.omega_WBs:
        assert isinstance(o, Expression)

    assert len(vars.p_c_Ws) == num_knot_points
    for p in vars.p_c_Ws:
        assert p.shape == (2, 1)

    assert len(vars.f_c_Ws) == num_knot_points
    for f in vars.f_c_Ws:
        assert f.shape == (2, 1)


if __name__ == "__main__":
    test_face_contact_variables()
