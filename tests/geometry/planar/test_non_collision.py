import numpy as np
import pydrake.symbolic as sym
from pydrake.solvers import Solve

from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    ContactLocation,
    PolytopeContactLocation,
)
from planning_through_contact.geometry.planar.non_collision import (
    NonCollisionMode,
    NonCollisionVariables,
)
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.planar.trajectory_builder import (
    PlanarTrajectoryBuilder,
)
from planning_through_contact.geometry.rigid_body import RigidBody
from planning_through_contact.planning.planar.planar_plan_specs import PlanarPlanSpecs
from planning_through_contact.planning.planar.tools import (
    check_finger_pose_in_contact_location,
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
from tests.geometry.planar.tools import (
    assert_initial_and_final_poses,
    assert_object_is_avoided,
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
    non_collision_mode.set_finger_initial_pose(finger_initial_pose)
    finger_final_pose = PlanarPose(-0.15, 0, 0)
    non_collision_mode.set_finger_final_pose(finger_final_pose)

    result = Solve(non_collision_mode.prog)
    assert result.is_success()

    vars = non_collision_mode.variables.eval_result(result)
    traj = PlanarTrajectoryBuilder([vars]).get_trajectory(interpolate=False)

    assert_initial_and_final_poses(
        traj, slider_pose, finger_initial_pose, slider_pose, finger_final_pose
    )

    DEBUG = False
    if DEBUG:
        visualize_planar_pushing_trajectory(traj, non_collision_mode.object.geometry)


def test_infeasible_non_collision_mode(non_collision_mode: NonCollisionMode) -> None:
    slider_pose = PlanarPose(0.3, 0, 0)
    non_collision_mode.set_slider_pose(slider_pose)

    finger_initial_pose = PlanarPose(-0.2, 0.1, 0)
    non_collision_mode.set_finger_initial_pose(finger_initial_pose)
    finger_final_pose = PlanarPose(0, 0, 0)  # Will cause penetration
    non_collision_mode.set_finger_final_pose(finger_final_pose)

    result = Solve(non_collision_mode.prog)
    assert not result.is_success()


def test_pos_in_loc(rigid_body_box: RigidBody) -> None:
    loc = PolytopeContactLocation(ContactLocation.FACE, 2)
    body_pose = PlanarPose(0, 0, 0)

    finger_pose_1 = PlanarPose(0, 0, 0)
    res_1 = check_finger_pose_in_contact_location(
        finger_pose_1, loc, rigid_body_box, body_pose
    )
    assert res_1 == False  # penetrates the box

    finger_pose_2 = PlanarPose(0, -0.6, 0)
    res_2 = check_finger_pose_in_contact_location(
        finger_pose_2, loc, rigid_body_box, body_pose
    )
    assert res_2 == True

    finger_pose_3 = PlanarPose(0.1, -0.6, 0)
    res_3 = check_finger_pose_in_contact_location(
        finger_pose_3, loc, rigid_body_box, body_pose
    )
    assert res_3 == True

    loc_2 = PolytopeContactLocation(ContactLocation.FACE, 3)
    body_pose_2 = PlanarPose(0.1, 0, 0)
    finger_pose_4 = PlanarPose(-0.4, 0, 0)
    res_4 = check_finger_pose_in_contact_location(
        finger_pose_4, loc_2, rigid_body_box, body_pose_2
    )
    assert res_4 == True


def test_multiple_knot_points(rigid_body_box: RigidBody) -> None:
    NUM_KNOT_POINTS = 5
    specs = PlanarPlanSpecs(
        num_knot_points_non_collision=NUM_KNOT_POINTS, time_non_collision=3
    )
    loc = PolytopeContactLocation(ContactLocation.FACE, 3)

    mode = NonCollisionMode.create_from_plan_spec(loc, specs, rigid_body_box)

    slider_pose = PlanarPose(0.3, 0, 0)
    mode.set_slider_pose(slider_pose)

    finger_initial_pose = PlanarPose(-0.2, 0.1, 0)
    mode.set_finger_initial_pose(finger_initial_pose)
    finger_final_pose = PlanarPose(-0.4, -0.2, 0)
    mode.set_finger_final_pose(finger_final_pose)

    result = Solve(mode.prog)
    assert result.is_success()

    vars = mode.variables.eval_result(result)
    traj = PlanarTrajectoryBuilder([vars]).get_trajectory(interpolate=False)

    assert_initial_and_final_poses(
        traj, slider_pose, finger_initial_pose, slider_pose, finger_final_pose
    )

    target_pos = [
        v.squeeze()
        for v in np.linspace(
            finger_initial_pose.pos(), finger_final_pose.pos(), num=NUM_KNOT_POINTS
        )
    ]  # t.shape = (2,)
    for p, t in zip(traj.p_c_B.T, target_pos):  # p.shape = (2,)
        assert np.allclose(p, t)

    DEBUG = False
    if DEBUG:
        visualize_planar_pushing_trajectory(traj, mode.object.geometry)


def test_avoid_object(rigid_body_box: RigidBody) -> None:
    NUM_KNOT_POINTS = 5
    specs = PlanarPlanSpecs(
        num_knot_points_non_collision=NUM_KNOT_POINTS, time_non_collision=3
    )
    loc = PolytopeContactLocation(ContactLocation.FACE, 3)

    mode = NonCollisionMode.create_from_plan_spec(
        loc, specs, rigid_body_box, avoid_object=True
    )

    assert len(mode.prog.quadratic_costs()) == 2

    slider_pose = PlanarPose(0.3, 0, 0)
    mode.set_slider_pose(slider_pose)

    finger_initial_pose = PlanarPose(-0.15, 0.13, 0)
    mode.set_finger_initial_pose(finger_initial_pose)
    finger_final_pose = PlanarPose(-0.15, -0.13, 0)
    mode.set_finger_final_pose(finger_final_pose)

    result = Solve(mode.prog)
    assert result.is_success()

    vars = mode.variables.eval_result(result)
    traj = PlanarTrajectoryBuilder([vars]).get_trajectory(interpolate=False)

    assert_initial_and_final_poses(
        traj, slider_pose, finger_initial_pose, slider_pose, finger_final_pose
    )

    assert_object_is_avoided(rigid_body_box.geometry, traj.p_c_B)

    DEBUG = False
    if DEBUG:
        visualize_planar_pushing_trajectory(traj, mode.object.geometry)
