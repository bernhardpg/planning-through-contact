# import pytest
from pydrake.solvers import MathematicalProgram
from pydrake.symbolic import Expression, Variables

from geometry.collision_geometry.box_2d import Box2d
from geometry.collision_geometry.collision_geometry import (
    ContactLocation,
    PolytopeContactLocation,
)
from geometry.planar.planar_contact_modes import (
    FaceContactMode,
    FaceContactVariables,
    PlanarPlanSpecs,
)
from geometry.rigid_body import RigidBody


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


def test_face_contact_mode() -> None:
    box_geometry = Box2d(width=0.1, height=0.2)
    mass = 0.1
    box = RigidBody("box", box_geometry, mass)
    contact_location = PolytopeContactLocation(
        ContactLocation.FACE, 3
    )  # We use the same face as in the T-pusher demo to make it simple to write tests
    specs = PlanarPlanSpecs()

    num_knot_points = 4
    time_in_contact = 2

    mode = FaceContactMode.create_from_plan_spec(contact_location, specs, box)

    # for each knot point:
    # 0 <= lam <= 0 and normal_force >= 0
    NUM_BBOX = 3
    assert len(mode.prog.bounding_box_constraints()) == num_knot_points * NUM_BBOX

    # for each finite difference knot point:
    # v_c_B == 0 and x and y quasi-static dynamics
    # TODO(bernhardpg): Will get fewer linear equality constraints once the wrench is rotated to the world frame
    NUM_LIN_EQS = 3
    assert (
        len(mode.prog.linear_equality_constraints())
        == (num_knot_points - 1) * NUM_LIN_EQS
    )


def test_quasi_static_dynamics() -> None:
    prog = MathematicalProgram()
    box_geometry = Box2d(width=0.2, height=0.1)
    mass = 0.1
    friction_coeff = 0.5
    contact_location = PolytopeContactLocation(ContactLocation.FACE, 3)

    num_knot_points = 4
    time_in_contact = 2

    vars = FaceContactVariables.from_prog(
        prog,
        box_geometry,
        contact_location,
        num_knot_points,
        time_in_contact,
    )

    k = 0

    f_c_B = vars.f_c_Bs[k]
    p_c_B = vars.p_c_Bs[k]
    R_WB = vars.R_WBs[k]
    v_WB = vars.v_WBs[k]
    omega_WB = vars.omega_WBs[k]

    x_dot, dyn = FaceContactMode.quasi_static_dynamics(
        v_WB, omega_WB, f_c_B, p_c_B, R_WB, friction_coeff, mass
    )

    check_vars_eq = lambda e, v: e.GetVariables().EqualTo(Variables(v))
    assert check_vars_eq(dyn[0], [vars.normal_forces[0]])
    assert check_vars_eq(dyn[1], [vars.friction_forces[0]])
    assert check_vars_eq(
        dyn[2], [vars.lams[0], vars.normal_forces[0], vars.friction_forces[0]]
    )


if __name__ == "__main__":
    # test_face_contact_variables()
    test_face_contact_mode()
    # test_quasi_static_dynamics()
