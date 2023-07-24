import pytest
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


@pytest.fixture
def box_geometry() -> Box2d:
    return Box2d(width=0.2, height=0.1)


@pytest.fixture
def face_contact_vars(box_geometry: Box2d) -> FaceContactVariables:
    prog = MathematicalProgram()
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
    return vars


def test_face_contact_variables(
    box_geometry: Box2d, face_contact_vars: FaceContactVariables
) -> None:
    num_knot_points = face_contact_vars.num_knot_points

    assert face_contact_vars.lams.shape == (num_knot_points,)
    assert face_contact_vars.normal_forces.shape == (num_knot_points,)
    assert face_contact_vars.friction_forces.shape == (num_knot_points,)
    assert face_contact_vars.cos_ths.shape == (num_knot_points,)
    assert face_contact_vars.sin_ths.shape == (num_knot_points,)
    assert face_contact_vars.p_WB_xs.shape == (num_knot_points,)
    assert face_contact_vars.p_WB_ys.shape == (num_knot_points,)

    assert all(face_contact_vars.pv1 == box_geometry.vertices[3])
    assert all(face_contact_vars.pv2 == box_geometry.vertices[0])

    assert len(face_contact_vars.R_WBs) == num_knot_points
    for R in face_contact_vars.R_WBs:
        assert R.shape == (2, 2)

    assert len(face_contact_vars.p_WBs) == num_knot_points
    for p in face_contact_vars.p_WBs:
        assert p.shape == (2, 1)

    assert len(face_contact_vars.f_c_Bs) == num_knot_points
    for f in face_contact_vars.f_c_Bs:
        assert f.shape == (2, 1)

    assert len(face_contact_vars.p_c_Bs) == num_knot_points
    for p_c in face_contact_vars.p_c_Bs:
        assert p_c.shape == (2, 1)

    assert len(face_contact_vars.v_WBs) == num_knot_points - 1
    for v in face_contact_vars.v_WBs:
        assert v.shape == (2, 1)

    assert len(face_contact_vars.cos_th_dots) == num_knot_points - 1
    for c in face_contact_vars.cos_th_dots:
        assert isinstance(c, Expression)

    assert len(face_contact_vars.sin_th_dots) == num_knot_points - 1
    for s in face_contact_vars.cos_th_dots:
        assert isinstance(s, Expression)

    assert len(face_contact_vars.v_c_Bs) == num_knot_points - 1
    for v in face_contact_vars.v_c_Bs:
        assert v.shape == (2, 1)

    assert len(face_contact_vars.omega_WBs) == num_knot_points - 1
    for o in face_contact_vars.omega_WBs:
        assert isinstance(o, Expression)

    assert len(face_contact_vars.p_c_Ws) == num_knot_points
    for p in face_contact_vars.p_c_Ws:
        assert p.shape == (2, 1)

    assert len(face_contact_vars.f_c_Ws) == num_knot_points
    for f in face_contact_vars.f_c_Ws:
        assert f.shape == (2, 1)


def test_face_contact_mode(box_geometry: Box2d) -> None:
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


def test_quasi_static_dynamics(face_contact_vars: FaceContactVariables) -> None:
    mass = 0.1
    friction_coeff = 0.5

    k = 0

    f_c_B = face_contact_vars.f_c_Bs[k]
    p_c_B = face_contact_vars.p_c_Bs[k]
    R_WB = face_contact_vars.R_WBs[k]
    v_WB = face_contact_vars.v_WBs[k]
    omega_WB = face_contact_vars.omega_WBs[k]

    _, dyn = FaceContactMode.quasi_static_dynamics(
        v_WB, omega_WB, f_c_B, p_c_B, R_WB, friction_coeff, mass
    )

    check_vars_eq = lambda e, v: e.GetVariables().EqualTo(Variables(v))
    assert check_vars_eq(dyn[0], [face_contact_vars.normal_forces[0]])
    assert check_vars_eq(dyn[1], [face_contact_vars.friction_forces[0]])
    assert check_vars_eq(
        dyn[2],
        [
            face_contact_vars.lams[0],
            face_contact_vars.normal_forces[0],
            face_contact_vars.friction_forces[0],
        ],
    )


# if __name__ == "__main__":
#     # test_face_contact_variables()
#     # test_face_contact_mode(box_geometry())
#     test_quasi_static_dynamics(face_contact_vars(box_geometry()))
