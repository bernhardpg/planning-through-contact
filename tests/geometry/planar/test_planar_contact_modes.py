import numpy as np
import pytest
from pydrake.solvers import MathematicalProgram, Solve
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
from geometry.planar.planar_pose import PlanarPose
from geometry.planar.trajectory_builder import PlanarTrajectoryBuilder
from geometry.rigid_body import RigidBody
from visualize.analysis import plot_cos_sine_trajs
from visualize.planar import visualize_planar_pushing_trajectory


@pytest.fixture
def box_geometry() -> Box2d:
    return Box2d(width=0.3, height=0.3)


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


@pytest.fixture
def face_contact_mode(box_geometry: Box2d) -> FaceContactMode:
    mass = 0.3
    box = RigidBody("box", box_geometry, mass)
    contact_location = PolytopeContactLocation(
        ContactLocation.FACE, 3
    )  # We use the same face as in the T-pusher demo to make it simple to write tests
    specs = PlanarPlanSpecs()
    mode = FaceContactMode.create_from_plan_spec(contact_location, specs, box)
    return mode


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


def test_face_contact_mode(face_contact_mode: FaceContactMode) -> None:
    mode = face_contact_mode
    num_knot_points = mode.num_knot_points

    # for each knot point:
    # 0 <= lam <= 0 and normal_force >= 0
    num_bbox = num_knot_points * 3
    assert len(mode.prog.bounding_box_constraints()) == num_bbox

    # for each finite difference knot point:
    # v_c_B == 0 and x and y quasi-static dynamics
    # TODO(bernhardpg): Will get fewer linear equality constraints once the wrench is rotated to the world frame
    num_lin_eq = (num_knot_points - 1) * 3
    assert len(mode.prog.linear_equality_constraints()) == num_lin_eq

    # for each knot point:
    # | c_f | <= \mu * c_n
    num_lin = num_knot_points * 2
    assert len(mode.prog.linear_constraints()) == num_lin

    # for each knot point:
    # c**2 + s**2 == 1
    # for each finite diff point:
    # omega_WB == torque_W
    num_quad = num_knot_points + (num_knot_points - 1)
    assert len(mode.prog.quadratic_constraints()) == num_quad

    tot_num_consts = num_bbox + num_lin_eq + num_lin + num_quad
    assert len(mode.prog.GetAllConstraints()) == tot_num_consts

    assert len(mode.prog.linear_costs()) == 0

    # One quadratic cost for linear and angular velocities
    assert len(mode.prog.quadratic_costs()) == 2

    lin_vel_vars = Variables(mode.prog.quadratic_costs()[0].variables())
    target_lin_vel_vars = Variables(np.concatenate(mode.variables.p_WBs))
    assert lin_vel_vars.EqualTo(target_lin_vel_vars)

    ang_vel_vars = Variables(mode.prog.quadratic_costs()[1].variables())
    target_ang_vel_vars = Variables(
        np.concatenate((mode.variables.cos_ths, mode.variables.sin_ths))
    )
    assert ang_vel_vars.EqualTo(target_ang_vel_vars)


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
    assert check_vars_eq(dyn[0, 0], [face_contact_vars.normal_forces[0]])
    assert check_vars_eq(dyn[1, 0], [face_contact_vars.friction_forces[0]])
    assert check_vars_eq(
        dyn[2, 0],
        [
            face_contact_vars.lams[0],
            face_contact_vars.normal_forces[0],
            face_contact_vars.friction_forces[0],
        ],
    )


def test_one_contact_mode(face_contact_mode: FaceContactMode) -> None:
    initial_pose = PlanarPose(0, 0, 0)
    final_pose = PlanarPose(0.3, 0, 0.8)
    face_contact_mode.set_slider_initial_pose(initial_pose)
    face_contact_mode.set_slider_final_pose(final_pose)

    face_contact_mode.formulate_convex_relaxation()
    result = Solve(face_contact_mode.relaxed_prog)
    assert result.is_success()

    vars = face_contact_mode.variables.eval_result(result)
    traj = PlanarTrajectoryBuilder([vars]).get_trajectory(interpolate=False)

    assert np.allclose(traj.R_WB[0], initial_pose.two_d_rot_matrix())
    assert np.allclose(traj.p_WB[:, 0:1], initial_pose.pos())

    assert np.allclose(traj.R_WB[-1], final_pose.two_d_rot_matrix())
    assert np.allclose(traj.p_WB[:, -1:-2], final_pose.pos())

    DEBUG = False
    if DEBUG:
        # (num_knot_points, 2): first col cosines, second col sines
        rs = np.vstack([R_WB[:, 0] for R_WB in traj.R_WB])
        plot_cos_sine_trajs(rs)
        visualize_planar_pushing_trajectory(traj, face_contact_mode.object.geometry)
