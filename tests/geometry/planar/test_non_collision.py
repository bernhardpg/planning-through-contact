import numpy as np
import pydrake.symbolic as sym
from pydrake.solvers import Solve

from planning_through_contact.geometry.planar.non_collision import (
    NonCollisionMode,
    NonCollisionVariables,
)
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.planar.trajectory_builder import (
    PlanarTrajectoryBuilder,
)
from planning_through_contact.visualize.planar import (
    visualize_planar_pushing_trajectory,
)
from tests.geometry.planar.fixtures import (
    box_geometry,
    non_collision_mode,
    non_collision_vars,
    rigid_body_box,
)


def test_non_collision_vars(non_collision_vars: NonCollisionVariables) -> None:
    num_knot_points = non_collision_vars.num_knot_points

    assert isinstance(non_collision_vars.cos_th, sym.Variable) or isinstance(
        non_collision_vars.cos_th, float
    )
    assert isinstance(non_collision_vars.cos_th, sym.Variable) or isinstance(
        non_collision_vars.sin_th, float
    )
    assert isinstance(non_collision_vars.cos_th, sym.Variable) or isinstance(
        non_collision_vars.p_WB_x, float
    )
    assert isinstance(non_collision_vars.cos_th, sym.Variable) or isinstance(
        non_collision_vars.p_WB_y, float
    )
    assert non_collision_vars.p_BF_xs.shape == (num_knot_points,)
    assert non_collision_vars.p_BF_ys.shape == (num_knot_points,)

    assert len(non_collision_vars.R_WBs) == num_knot_points
    for R in non_collision_vars.R_WBs:
        assert R.shape == (2, 2)

    assert len(non_collision_vars.p_WBs) == num_knot_points
    for p in non_collision_vars.p_WBs:
        assert p.shape == (2, 1)

    assert len(non_collision_vars.p_BFs) == num_knot_points
    for p_BF in non_collision_vars.p_BFs:
        assert p_BF.shape == (2, 1)

    assert len(non_collision_vars.v_WBs) == num_knot_points - 1
    for v in non_collision_vars.v_WBs:
        assert v.shape == (2, 1)

    assert len(non_collision_vars.omega_WBs) == num_knot_points - 1
    for o in non_collision_vars.omega_WBs:
        assert isinstance(o, float)

    assert len(non_collision_vars.p_c_Ws) == num_knot_points
    for p in non_collision_vars.p_c_Ws:
        assert p.shape == (2, 1)
        assert isinstance(p[0, 0], sym.Expression)

    assert len(non_collision_vars.f_c_Ws) == num_knot_points
    for f in non_collision_vars.f_c_Ws:
        assert f.shape == (2, 1)
        assert np.all(f == 0)


def test_non_collision_mode(non_collision_mode: NonCollisionMode) -> None:
    mode = non_collision_mode
    num_knot_points = mode.num_knot_points

    # We should have three planes for a collision free region for a normal box
    num_planes = len(mode.planes)
    assert num_planes == 3

    # One linear constraint per plane, per knot point
    num_linear_constraints = len(mode.prog.linear_constraints()) + len(
        mode.prog.bounding_box_constraints()
    )
    assert num_linear_constraints == num_knot_points * num_planes

    # The next two tests may fail for more complex geometries than boxes. If so, update them!
    assert len(mode.prog.bounding_box_constraints()) == 2
    assert len(mode.prog.linear_constraints()) == 4

    assert len(mode.prog.linear_equality_constraints()) == 0

    assert len(mode.prog.linear_costs()) == 0

    # One quadratic cost for squared eucl distances
    assert len(mode.prog.quadratic_costs()) == 1

    lin_vel_vars = sym.Variables(mode.prog.quadratic_costs()[0].variables())
    target_lin_vel_vars = sym.Variables(np.concatenate(mode.variables.p_BFs))
    assert lin_vel_vars.EqualTo(target_lin_vel_vars)


def test_one_non_collision_mode(non_collision_mode: NonCollisionMode) -> None:
    slider_pose = PlanarPose(0.3, 0, 0)
    non_collision_mode.set_slider_pose(slider_pose)

    finger_initial_pose = PlanarPose(-0.2, 0.1, 0)
    non_collision_mode.set_finger_initial_pos(finger_initial_pose.pos())
    finger_final_pose = PlanarPose(-0.15, 0, 0)
    non_collision_mode.set_finger_final_pos(finger_final_pose.pos())

    result = Solve(non_collision_mode.prog)
    assert result.is_success()

    vars = non_collision_mode.variables.eval_result(result)
    traj = PlanarTrajectoryBuilder([vars]).get_trajectory(interpolate=False)

    assert np.allclose(traj.R_WB[0], slider_pose.two_d_rot_matrix())
    assert np.allclose(traj.p_WB[:, 0:1], slider_pose.pos())

    assert np.allclose(traj.R_WB[-1], slider_pose.two_d_rot_matrix())
    assert np.allclose(traj.p_WB[:, -1:1], slider_pose.pos())

    assert np.allclose(
        traj.p_c_W[:, 0:1], slider_pose.pos() + finger_initial_pose.pos()
    )
    assert np.allclose(traj.p_c_W[:, -1:], slider_pose.pos() + finger_final_pose.pos())

    DEBUG = False
    if DEBUG:
        visualize_planar_pushing_trajectory(traj, non_collision_mode.object.geometry)


def test_infeasible_non_collision_mode(non_collision_mode: NonCollisionMode) -> None:
    slider_pose = PlanarPose(0.3, 0, 0)
    non_collision_mode.set_slider_pose(slider_pose)

    finger_initial_pose = PlanarPose(-0.2, 0.1, 0)
    non_collision_mode.set_finger_initial_pos(finger_initial_pose.pos())
    finger_final_pose = PlanarPose(0, 0, 0)  # Will cause penetration
    non_collision_mode.set_finger_final_pos(finger_final_pose.pos())

    result = Solve(non_collision_mode.prog)
    assert not result.is_success()
