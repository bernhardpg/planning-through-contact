from typing import Literal

import numpy as np
import pydrake.symbolic as sym
import pytest
from pydrake.solvers import Solve

from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    ContactLocation,
    PolytopeContactLocation,
)
from planning_through_contact.geometry.collision_geometry.t_pusher_2d import TPusher2d
from planning_through_contact.geometry.hyperplane import Hyperplane
from planning_through_contact.geometry.planar.non_collision import (
    NonCollisionMode,
    NonCollisionVariables,
    check_finger_pose_in_contact_location,
)
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.planar.trajectory_builder import (
    PlanarTrajectoryBuilder,
)
from planning_through_contact.geometry.rigid_body import RigidBody
from planning_through_contact.planning.planar.planar_plan_config import (
    PlanarPlanConfig,
    SliderPusherSystemConfig,
)
from planning_through_contact.visualize.planar_pushing import (
    visualize_planar_pushing_trajectory_legacy,
)
from tests.geometry.planar.fixtures import (
    box_geometry,
    dynamics_config,
    non_collision_mode,
    non_collision_vars,
    plan_config,
    rigid_body_box,
)
from tests.geometry.planar.tools import (
    assert_initial_and_final_poses_LEGACY,
    assert_object_is_avoided,
)

DEBUG = False


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
    assert non_collision_vars.p_BP_xs.shape == (num_knot_points,)
    assert non_collision_vars.p_BP_ys.shape == (num_knot_points,)

    assert len(non_collision_vars.R_WBs) == num_knot_points
    for R in non_collision_vars.R_WBs:
        assert R.shape == (2, 2)

    assert len(non_collision_vars.p_WBs) == num_knot_points
    for p in non_collision_vars.p_WBs:
        assert p.shape == (2, 1)

    assert len(non_collision_vars.p_BPs) == num_knot_points
    for p_BF in non_collision_vars.p_BPs:
        assert p_BF.shape == (2, 1)

    assert len(non_collision_vars.v_WBs) == num_knot_points - 1
    for v in non_collision_vars.v_WBs:
        assert v.shape == (2, 1)

    assert len(non_collision_vars.omega_WBs) == num_knot_points - 1
    for o in non_collision_vars.omega_WBs:
        assert isinstance(o, float)

    assert len(non_collision_vars.p_WPs) == num_knot_points
    for p in non_collision_vars.p_WPs:
        assert p.shape == (2, 1)
        assert isinstance(p[0, 0], sym.Expression)

    assert len(non_collision_vars.f_c_Ws) == num_knot_points - 1
    for f in non_collision_vars.f_c_Ws:
        assert f.shape == (2, 1)
        assert np.all(f == 0)


def test_non_collision_mode(non_collision_mode: NonCollisionMode) -> None:
    mode = non_collision_mode
    num_knot_points = mode.num_knot_points

    # We should have two planes for a collision free region for a normal box
    num_planes = len(mode.collision_free_space_planes)
    assert num_planes == 2

    assert isinstance(mode.contact_planes[0], Hyperplane)

    num_linear_constraints = len(mode.prog.linear_constraints()) + len(
        mode.prog.bounding_box_constraints()
    )
    num_planes = 3
    expected_num_lin_consts = num_knot_points * num_planes
    assert num_linear_constraints == expected_num_lin_consts

    # 2 dimensions, 3 planes, i.e. 6 ineq/bbox constraints (depending on geometry)
    assert (
        len(mode.prog.bounding_box_constraints()) + len(mode.prog.linear_constraints())
        == 6
    )

    assert len(mode.prog.linear_equality_constraints()) == 0

    assert len(mode.prog.linear_costs()) == 0

    # One quadratic cost for squared eucl distances
    assert len(mode.prog.quadratic_costs()) == 1

    lin_vel_vars = sym.Variables(mode.prog.quadratic_costs()[0].variables())
    target_lin_vel_vars = sym.Variables(np.concatenate(mode.variables.p_BPs))
    assert lin_vel_vars.EqualTo(target_lin_vel_vars)


def test_one_non_collision_mode(non_collision_mode: NonCollisionMode) -> None:
    slider_pose = PlanarPose(0.3, 0.3, 0)
    non_collision_mode.set_slider_pose(slider_pose)

    finger_initial_pose = PlanarPose(-0.2, 0.1, 0)
    non_collision_mode.set_finger_initial_pose(finger_initial_pose)
    finger_final_pose = PlanarPose(-0.18, 0, 0)
    non_collision_mode.set_finger_final_pose(finger_final_pose)

    result = Solve(non_collision_mode.prog)
    assert result.is_success()

    vars = non_collision_mode.variables.eval_result(result)
    traj = PlanarTrajectoryBuilder([vars]).get_trajectory(interpolate=False)

    assert_initial_and_final_poses_LEGACY(
        traj, slider_pose, finger_initial_pose, slider_pose, finger_final_pose
    )

    if DEBUG:
        visualize_planar_pushing_trajectory_legacy(
            traj,
            non_collision_mode.object.geometry,
            pusher_radius=non_collision_mode.pusher_radius,
        )


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

    config = PlanarPlanConfig(
        dynamics_config=SliderPusherSystemConfig(slider=rigid_body_box)
    )

    finger_pose_1 = PlanarPose(0, 0, 0)
    res_1 = check_finger_pose_in_contact_location(finger_pose_1, loc, config)
    assert res_1 == False  # penetrates the box

    finger_pose_2 = PlanarPose(0, -0.3, 0)
    res_2 = check_finger_pose_in_contact_location(finger_pose_2, loc, config)
    assert res_2 == True

    finger_pose_3 = PlanarPose(0.1, -0.3, 0)
    res_3 = check_finger_pose_in_contact_location(finger_pose_3, loc, config)
    assert res_3 == True

    loc_2 = PolytopeContactLocation(ContactLocation.FACE, 3)
    finger_pose_4 = PlanarPose(-0.2, 0, 0)
    res_4 = check_finger_pose_in_contact_location(finger_pose_4, loc_2, config)
    assert res_4 == True


def test_eucl_dist(plan_config: PlanarPlanConfig) -> None:
    NUM_KNOT_POINTS = 3
    plan_config.num_knot_points_non_collision = NUM_KNOT_POINTS
    plan_config.minimize_squared_eucl_dist = False
    loc = PolytopeContactLocation(ContactLocation.FACE, 3)

    mode = NonCollisionMode.create_from_plan_spec(loc, plan_config)

    assert len(mode.prog.linear_costs()) == NUM_KNOT_POINTS - 1
    assert len(mode.prog.quadratic_costs()) == 0


def test_multiple_knot_points(plan_config: PlanarPlanConfig) -> None:
    NUM_KNOT_POINTS = 5
    plan_config.num_knot_points_non_collision = NUM_KNOT_POINTS
    plan_config.dynamics_config.pusher_radius = 0.03
    loc = PolytopeContactLocation(ContactLocation.FACE, 3)

    mode = NonCollisionMode.create_from_plan_spec(loc, plan_config)

    slider_pose = PlanarPose(0.3, 0.3, 0)
    mode.set_slider_pose(slider_pose)

    finger_initial_pose = PlanarPose(-0.2, 0.1, 0)
    mode.set_finger_initial_pose(finger_initial_pose)
    finger_final_pose = PlanarPose(-0.4, -0.2, 0)
    mode.set_finger_final_pose(finger_final_pose)

    result = Solve(mode.prog)
    assert result.is_success()

    vars = mode.variables.eval_result(result)
    traj = PlanarTrajectoryBuilder([vars]).get_trajectory(interpolate=False)

    assert_initial_and_final_poses_LEGACY(
        traj, slider_pose, finger_initial_pose, slider_pose, finger_final_pose
    )

    target_pos = [
        v.squeeze()
        for v in np.linspace(
            finger_initial_pose.pos(), finger_final_pose.pos(), num=NUM_KNOT_POINTS
        )
    ]  # t.shape = (2,)
    for p, t in zip(traj.p_BP.T, target_pos):  # p.shape = (2,)
        assert np.allclose(p, t)

    if DEBUG:
        visualize_planar_pushing_trajectory_legacy(
            traj, plan_config.slider_geometry, pusher_radius=plan_config.pusher_radius
        )


# TODO(bernhardpg): remove this, we want to remove both the quadratic and linear objectives!
def test_avoid_object_quadratic(plan_config: PlanarPlanConfig) -> None:
    NUM_KNOT_POINTS = 5
    plan_config.num_knot_points_non_collision = NUM_KNOT_POINTS
    plan_config.dynamics_config.pusher_radius = 0.03
    plan_config.avoid_object = True
    loc = PolytopeContactLocation(ContactLocation.FACE, 3)

    mode = NonCollisionMode.create_from_plan_spec(loc, plan_config)

    assert len(mode.prog.quadratic_costs()) == 2

    slider_pose = PlanarPose(0.3, 0.3, 0)
    mode.set_slider_pose(slider_pose)

    finger_initial_pose = PlanarPose(-0.2, 0.1, 0)
    mode.set_finger_initial_pose(finger_initial_pose)
    finger_final_pose = PlanarPose(-0.2, -0.2, 0)
    mode.set_finger_final_pose(finger_final_pose)

    result = Solve(mode.prog)
    assert result.is_success()

    vars = mode.variables.eval_result(result)
    traj = PlanarTrajectoryBuilder([vars]).get_trajectory(interpolate=False)

    assert_initial_and_final_poses_LEGACY(
        traj,
        slider_pose,
        finger_initial_pose,
        slider_pose,
        finger_final_pose,
    )

    assert_object_is_avoided(plan_config.slider_geometry, traj.p_BP)

    # Pusher should move away from object
    assert vars.p_BP_xs[2] <= -0.27

    if DEBUG:
        visualize_planar_pushing_trajectory_legacy(
            traj, plan_config.slider_geometry, pusher_radius=plan_config.pusher_radius
        )


def test_avoid_object_socp(plan_config: PlanarPlanConfig) -> None:
    NUM_KNOT_POINTS = 5
    plan_config.num_knot_points_non_collision = NUM_KNOT_POINTS
    plan_config.dynamics_config.pusher_radius = 0.03
    plan_config.avoid_object = True
    plan_config.avoidance_cost = "socp_single_mode"
    loc = PolytopeContactLocation(ContactLocation.FACE, 3)

    mode = NonCollisionMode.create_from_plan_spec(loc, plan_config)

    slider_pose = PlanarPose(0.3, 0.3, 0)
    mode.set_slider_pose(slider_pose)

    finger_initial_pose = PlanarPose(-0.2, 0.1, 0)
    mode.set_finger_initial_pose(finger_initial_pose)
    finger_final_pose = PlanarPose(-0.2, -0.2, 0)
    mode.set_finger_final_pose(finger_final_pose)

    result = Solve(mode.prog)
    assert result.is_success()

    vars = mode.variables.eval_result(result)
    traj = PlanarTrajectoryBuilder([vars]).get_trajectory(interpolate=False)

    assert_initial_and_final_poses_LEGACY(
        traj,
        slider_pose,
        finger_initial_pose,
        slider_pose,
        finger_final_pose,
    )

    assert_object_is_avoided(plan_config.slider_geometry, traj.p_BP)

    # Pusher should move away from object
    assert vars.p_BP_xs[2] <= -0.25

    if DEBUG:
        visualize_planar_pushing_trajectory_legacy(
            traj, plan_config.slider_geometry, plan_config.pusher_radius
        )


@pytest.mark.parametrize(
    "loc, initial, target, cost",
    [
        (
            PolytopeContactLocation(ContactLocation.FACE, 0),
            PlanarPose(-0.1, 0.1, 0),
            PlanarPose(0.1, 0.1, 0),
            "quadratic",
        ),
        (
            PolytopeContactLocation(ContactLocation.FACE, 2),
            PlanarPose(0.05, -0.06, 0),
            PlanarPose(0.1, -0.06, 0),
            "quadratic",
        ),
        (
            PolytopeContactLocation(ContactLocation.FACE, 2),
            PlanarPose(0.05, -0.06, 0),
            PlanarPose(0.1, -0.06, 0),
            "socp_single_mode",
        ),
    ],
    ids=[1, 2, 3],
)
def test_avoid_object_t_pusher(
    plan_config: PlanarPlanConfig,
    loc: PolytopeContactLocation,
    initial: PlanarPose,
    target: PlanarPose,
    cost: Literal["quadratic", "socp_single_mode"],
) -> None:
    plan_config.num_knot_points_non_collision = 5
    plan_config.dynamics_config.pusher_radius = 0.015
    plan_config.avoid_object = True
    plan_config.avoidance_cost = cost
    plan_config.time_non_collision = 3
    plan_config.dynamics_config.slider = RigidBody("T", TPusher2d(), mass=0.2)

    mode = NonCollisionMode.create_from_plan_spec(loc, plan_config)

    slider_pose = PlanarPose(0.3, 0.3, 0)
    mode.set_slider_pose(slider_pose)

    mode.set_finger_initial_pose(initial)
    mode.set_finger_final_pose(target)

    result = Solve(mode.prog)
    assert result.is_success()

    vars = mode.variables.eval_result(result)
    traj = PlanarTrajectoryBuilder([vars]).get_trajectory(interpolate=False)

    assert_initial_and_final_poses_LEGACY(
        traj,
        slider_pose,
        initial,
        slider_pose,
        target,
    )

    assert_object_is_avoided(plan_config.slider_geometry, traj.p_BP)

    if DEBUG:
        visualize_planar_pushing_trajectory_legacy(
            traj, plan_config.slider_geometry, plan_config.pusher_radius
        )
