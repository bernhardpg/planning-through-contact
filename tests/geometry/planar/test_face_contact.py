import numpy as np
import pytest
from pydrake.solvers import MosekSolver
from pydrake.symbolic import Expression, Variables

from planning_through_contact.convex_relaxation.sdp import (
    eliminate_equality_constraints,
)
from planning_through_contact.geometry.collision_geometry.box_2d import Box2d
from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    ContactLocation,
    PolytopeContactLocation,
)
from planning_through_contact.geometry.planar.face_contact import (
    FaceContactMode,
    FaceContactVariables,
)
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.planar.planar_pushing_trajectory import (
    PlanarPushingTrajectory,
)
from planning_through_contact.geometry.planar.trajectory_builder import (
    PlanarTrajectoryBuilder,
)
from planning_through_contact.geometry.rigid_body import RigidBody
from planning_through_contact.planning.planar.planar_plan_config import (
    PlanarPlanConfig,
    PlanarPushingStartAndGoal,
    SliderPusherSystemConfig,
)
from planning_through_contact.tools.utils import evaluate_np_expressions_array
from planning_through_contact.visualize.analysis import plot_cos_sine_trajs
from planning_through_contact.visualize.planar_pushing import (
    visualize_planar_pushing_trajectory,
    visualize_planar_pushing_trajectory_legacy,
)
from tests.geometry.planar.fixtures import (
    box_geometry,
    face_contact_mode,
    face_contact_vars,
    plan_config,
    rigid_body_box,
    t_pusher,
)
from tests.geometry.planar.tools import assert_initial_and_final_poses

DEBUG = False


def test_face_contact_variables(
    box_geometry: Box2d, face_contact_vars: FaceContactVariables
) -> None:
    num_knot_points = face_contact_vars.num_knot_points

    assert face_contact_vars.lams.shape == (num_knot_points,)
    assert face_contact_vars.normal_forces.shape == (num_knot_points - 1,)
    assert face_contact_vars.friction_forces.shape == (num_knot_points - 1,)
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

    assert len(face_contact_vars.f_c_Bs) == num_knot_points - 1
    for f in face_contact_vars.f_c_Bs:
        assert f.shape == (2, 1)

    assert len(face_contact_vars.p_BPs) == num_knot_points
    for p_c in face_contact_vars.p_BPs:
        assert p_c.shape == (2, 1)

    assert len(face_contact_vars.v_WBs) == num_knot_points - 1
    for v in face_contact_vars.v_WBs:
        assert v.shape == (2, 1)

    assert len(face_contact_vars.cos_th_vels) == num_knot_points - 1
    for c in face_contact_vars.cos_th_vels:
        assert isinstance(c, Expression)

    assert len(face_contact_vars.sin_th_vels) == num_knot_points - 1
    for s in face_contact_vars.sin_th_vels:
        assert isinstance(s, Expression)

    assert len(face_contact_vars.v_BPs) == num_knot_points - 1
    for v in face_contact_vars.v_BPs:
        assert v.shape == (2, 1)

    assert len(face_contact_vars.delta_omega_WBs) == num_knot_points - 1
    for o in face_contact_vars.delta_omega_WBs:
        assert isinstance(o, Expression)


@pytest.mark.skip(
    reason="Stopped supported equation elimination so this test will fail"
)
def test_reduced_face_contact_variables(face_contact_mode: FaceContactMode) -> None:
    original_prog = face_contact_mode.prog_wrapper
    reduced_prog, get_x = eliminate_equality_constraints(original_prog)
    vars = face_contact_mode.variables

    reduced_vars = vars.from_reduced_prog(original_prog, reduced_prog, get_x)

    assert vars.lams.shape == reduced_vars.lams.shape
    assert vars.cos_ths.shape == reduced_vars.cos_ths.shape
    assert vars.sin_ths.shape == reduced_vars.sin_ths.shape
    assert vars.normal_forces.shape == reduced_vars.normal_forces.shape
    assert vars.friction_forces.shape == reduced_vars.friction_forces.shape
    assert vars.p_WB_xs.shape == reduced_vars.p_WB_xs.shape
    assert vars.p_WB_ys.shape == reduced_vars.p_WB_ys.shape


def test_face_contact_mode(face_contact_mode: FaceContactMode) -> None:
    mode = face_contact_mode
    num_knot_points = mode.num_knot_points
    prog = mode.prog_wrapper.prog

    # NOTE(bernhardpg): These are commented out, as we are currently not using bbox constraints
    # (they slow down the solution times a lot.
    # TODO(bernhardpg): This should be properly investigated

    # for each knot point:
    # each variable should have a bounding box constraint
    # lam, c_n, c_f, cos_th, sin_th, p_WB_x, p_WB_y
    # num_bbox = num_knot_points * 7
    # assert len(prog.bounding_box_constraints()) == num_bbox

    # NOTE(bernhardpg): With the current setup, we will have one bounding box constraint for the
    # 3 state vars, 1 input var
    num_bbox = 3 * num_knot_points + (num_knot_points - 1)
    assert len(prog.bounding_box_constraints()) == num_bbox

    # for each finite difference knot point:
    # v_c_B == 0
    num_lin_eq = 1 * (num_knot_points - 1)
    assert len(prog.linear_equality_constraints()) == num_lin_eq

    # for each knot point:
    # | c_f | <= \mu * c_n
    num_lin = (num_knot_points - 1) * 2
    assert len(prog.linear_constraints()) == num_lin

    # for each knot point:
    # c**2 + s**2 == 1
    # for each finite diff point:
    # quasi_static_dynamics (5 constraints)
    num_quad = num_knot_points + (num_knot_points - 1) * 5
    assert len(prog.quadratic_constraints()) == num_quad

    tot_num_consts = num_bbox + num_lin_eq + num_lin + num_quad
    assert len(prog.GetAllConstraints()) == tot_num_consts

    # Time in contact cost
    assert len(prog.linear_costs()) == 1

    # force regularization (2 per knot point - 1) and keypoint velocity regularization (per vertex per knot point - 1)
    assert len(prog.quadratic_costs()) == (mode.num_knot_points - 1) * (
        2 + len(mode.config.slider_geometry.vertices)
    )


@pytest.mark.parametrize(
    "face_contact_mode",
    [
        ({"face_idx": 3}),
        ({"face_idx": 0}),
    ],
    indirect=["face_contact_mode"],
    ids=["right", "loose"],
)
def test_one_contact_mode(face_contact_mode: FaceContactMode) -> None:
    initial_pose = PlanarPose(0, 0, 0)
    final_pose = PlanarPose(0.3, 0.2, 0.8)
    face_contact_mode.set_slider_initial_pose(initial_pose)
    face_contact_mode.set_slider_final_pose(final_pose)

    face_contact_mode.config.start_and_goal = PlanarPushingStartAndGoal(
        initial_pose, final_pose
    )

    face_contact_mode.formulate_convex_relaxation()
    solver = MosekSolver()
    result = solver.Solve(face_contact_mode.relaxed_prog)  # type: ignore
    assert result.is_success()

    vars = face_contact_mode.variables.eval_result(result)
    traj = PlanarPushingTrajectory(face_contact_mode.config, [vars])

    assert_initial_and_final_poses(traj, initial_pose, None, final_pose, None)

    if DEBUG:
        vars = face_contact_mode.variables.eval_result(result)
        traj = PlanarPushingTrajectory(face_contact_mode.config, [vars])

        visualize_planar_pushing_trajectory(
            traj, visualize_knot_points=True, save=True, filename="debug_file"
        )
        # (num_knot_points, 2): first col cosines, second col sines
        rs = np.vstack([R_WB[:, 0] for R_WB in traj.path_knot_points[0].R_WBs])  # type: ignore
        plot_cos_sine_trajs(rs)


@pytest.mark.parametrize(
    "face_contact_mode",
    [{"face_idx": 3}],
    indirect=["face_contact_mode"],
)
def test_one_contact_mode_minimize_keypoints(
    face_contact_mode: FaceContactMode,
) -> None:
    initial_pose = PlanarPose(0, 0, 0)
    final_pose = PlanarPose(0.3, 0.2, 0.8)
    face_contact_mode.set_slider_initial_pose(initial_pose)
    face_contact_mode.set_slider_final_pose(final_pose)

    face_contact_mode.config.start_and_goal = PlanarPushingStartAndGoal(
        initial_pose, final_pose
    )

    face_contact_mode.formulate_convex_relaxation()
    solver = MosekSolver()
    result = solver.Solve(face_contact_mode.relaxed_prog)  # type: ignore
    assert result.is_success()

    vars = face_contact_mode.variables.eval_result(result)
    traj = PlanarPushingTrajectory(face_contact_mode.config, [vars])

    assert_initial_and_final_poses(traj, initial_pose, None, final_pose, None)

    if DEBUG:
        vars = face_contact_mode.variables.eval_result(result)
        traj = PlanarPushingTrajectory(face_contact_mode.config, [vars])

        visualize_planar_pushing_trajectory(
            traj, visualize_knot_points=True, save=True, filename="debug_file"
        )
        # (num_knot_points, 2): first col cosines, second col sines
        rs = np.vstack([R_WB[:, 0] for R_WB in traj.path_knot_points[0].R_WBs])  # type: ignore
        plot_cos_sine_trajs(rs)


@pytest.mark.parametrize(
    "face_contact_mode",
    [
        ({"face_idx": 0, "body": "t_pusher"}),
        ({"face_idx": 3, "body": "t_pusher"}),
    ],
    indirect=["face_contact_mode"],
    ids=["tight", "loose"],
)
def test_planning_for_t_pusher(face_contact_mode: FaceContactMode) -> None:
    mode = face_contact_mode
    initial_pose = PlanarPose(0, 0, 0)
    final_pose = PlanarPose(0.3, 0, 0.1)
    mode.set_slider_initial_pose(initial_pose)
    mode.set_slider_final_pose(final_pose)

    face_contact_mode.config.start_and_goal = PlanarPushingStartAndGoal(
        initial_pose, final_pose
    )

    mode.formulate_convex_relaxation()
    result = MosekSolver().Solve(mode.relaxed_prog)  # type: ignore
    assert result.is_success()  # should fail when the relaxation is tight!

    vars = face_contact_mode.variables.eval_result(result)
    traj = PlanarPushingTrajectory(face_contact_mode.config, [vars])

    if DEBUG:
        visualize_planar_pushing_trajectory_legacy(
            traj, face_contact_mode.dynamics_config.slider.geometry, 0.01
        )
        # (num_knot_points, 2): first col cosines, second col sines
        rs = np.vstack([R_WB[:, 0] for R_WB in traj.R_WB])
        plot_cos_sine_trajs(rs)


@pytest.mark.parametrize(
    "face_contact_mode",
    [
        # TODO(bernhardpg): Get back to this after figuring out integration scheme
        # only infeasible with bounds, redundant dynamics constraints, and CONSTANT contact forces
        # (
        #     {"face_idx": 1}
        # ),
        # ({"face_idx": 0}),  # not infeasible, although it seems it should be?
    ],
    indirect=["face_contact_mode"],
)
def test_one_contact_mode_infeasible(face_contact_mode: FaceContactMode) -> None:
    initial_pose = PlanarPose(0, 0, 0)
    final_pose = PlanarPose(0.3, 0, 0.8)
    face_contact_mode.set_slider_initial_pose(initial_pose)
    face_contact_mode.set_slider_final_pose(final_pose)

    face_contact_mode.config.start_and_goal = PlanarPushingStartAndGoal(
        initial_pose, final_pose
    )

    face_contact_mode.formulate_convex_relaxation()
    result = MosekSolver().Solve(face_contact_mode.relaxed_prog)  # type: ignore
    assert not result.is_success()  # should fail when the relaxation is tight!

    if DEBUG:
        vars = face_contact_mode.variables.eval_result(result)
        traj = PlanarPushingTrajectory(face_contact_mode.config, [vars])
        visualize_planar_pushing_trajectory_legacy(
            traj, face_contact_mode.dynamics_config.slider.geometry, 0.01
        )
        # (num_knot_points, 2): first col cosines, second col sines
        rs = np.vstack([R_WB[:, 0] for R_WB in traj.R_WB])
        plot_cos_sine_trajs(rs)


@pytest.mark.parametrize(
    "face_contact_mode",
    [
        # TODO(bernhardpg): Get back to this after figuring out integration scheme
        # only infeasible with bounds, redundant dynamics constraints, and CONSTANT contact forces
        # ({"face_idx": 4, "body": "t_pusher"}),
        # ({"face_idx": 6, "body": "t_pusher"}),
    ],
    indirect=["face_contact_mode"],
)
def test_planning_for_t_pusher_infeasible(face_contact_mode: FaceContactMode) -> None:
    mode = face_contact_mode
    initial_pose = PlanarPose(0, 0, 0)
    final_pose = PlanarPose(0.3, 0, 0.3)
    mode.set_slider_initial_pose(initial_pose)
    mode.set_slider_final_pose(final_pose)

    face_contact_mode.config.start_and_goal = PlanarPushingStartAndGoal(
        initial_pose, final_pose
    )

    mode.formulate_convex_relaxation()
    result = MosekSolver().Solve(mode.relaxed_prog)  # type: ignore
    # assert not result.is_success()  # should fail when the relaxation is tight!

    if DEBUG:
        vars = face_contact_mode.variables.eval_result(result)
        traj = PlanarTrajectoryBuilder([vars]).get_trajectory(interpolate=False)
        visualize_planar_pushing_trajectory_legacy(
            traj, face_contact_mode.dynamics_config.slider.geometry, 0.01
        )
        # (num_knot_points, 2): first col cosines, second col sines
        rs = np.vstack([R_WB[:, 0] for R_WB in traj.R_WB])
        plot_cos_sine_trajs(rs)


@pytest.mark.skip(
    reason="Stopped supported equation elimination so this test will fail"
)
@pytest.mark.parametrize(
    "face_contact_mode",
    [
        {"face_idx": 3, "use_eq_elimination": True},
        # not infeasible, although it should be (relaxation is loose)
        {"face_idx": 0, "use_eq_elimination": True},
        {"face_idx": 0, "use_eq_elimination": True, "body": "t_pusher"},
        # not infeasible, although it should be (relaxation is loose)
        {"face_idx": 1, "use_eq_elimination": True, "body": "t_pusher"},
    ],
    indirect=["face_contact_mode"],
    ids=["box", "box_loose", "t_pusher", "t_pusher_loose"],
)
def test_face_contact_equality_elimination(face_contact_mode: FaceContactMode) -> None:
    initial_pose = PlanarPose(0, 0, 0)
    final_pose = PlanarPose(0.3, 0, 0.1)
    face_contact_mode.set_slider_initial_pose(initial_pose)
    face_contact_mode.set_slider_final_pose(final_pose)

    face_contact_mode.formulate_convex_relaxation()

    # now = time.time()
    result = MosekSolver().Solve(face_contact_mode.relaxed_prog)  # type: ignore
    # solve_time = time.time() - now

    # SOLVE_TIME_TRESHOLD = 0.08
    # assert solve_time <= SOLVE_TIME_TRESHOLD
    # Empirically all of these should solve within the treshold

    assert result.is_success()

    if DEBUG:
        vars = face_contact_mode.reduced_variables.eval_result(result)
        traj = PlanarTrajectoryBuilder([vars]).get_trajectory(interpolate=False)
        visualize_planar_pushing_trajectory_legacy(
            traj, face_contact_mode.dynamics_config.slider.geometry, 0.01
        )
        # (num_knot_points, 2): first col cosines, second col sines
        rs = np.vstack([R_WB[:, 0] for R_WB in traj.R_WB])
        plot_cos_sine_trajs(rs)


def test_get_X(rigid_body_box: RigidBody) -> None:
    face_idx = 3
    contact_location = PolytopeContactLocation(ContactLocation.FACE, face_idx)
    dynamics_config = SliderPusherSystemConfig(slider=rigid_body_box, pusher_radius=0.0)
    cfg = PlanarPlanConfig(
        dynamics_config=dynamics_config,
        use_approx_exponential_map=False,
        use_band_sparsity=False,
    )
    mode = FaceContactMode.create_from_plan_spec(contact_location, cfg)

    initial_pose = PlanarPose(0, 0, 0)
    final_pose = PlanarPose(0.3, 0, 0.8)
    mode.set_slider_initial_pose(initial_pose)
    mode.set_slider_final_pose(final_pose)

    mode.formulate_convex_relaxation()
    X = mode.get_Xs()[0]

    result = MosekSolver().Solve(mode.relaxed_prog)  # type: ignore
    X_sol = result.GetSolution(X)
    assert X.shape == (27, 27)
    assert X_sol.shape == (27, 27)


def test_get_X_band_sparse(rigid_body_box: RigidBody) -> None:
    face_idx = 3
    contact_location = PolytopeContactLocation(ContactLocation.FACE, face_idx)
    dynamics_config = SliderPusherSystemConfig(slider=rigid_body_box, pusher_radius=0.0)
    cfg = PlanarPlanConfig(
        dynamics_config=dynamics_config,
        use_approx_exponential_map=False,
        use_band_sparsity=True,
    )
    mode = FaceContactMode.create_from_plan_spec(contact_location, cfg)

    initial_pose = PlanarPose(0, 0, 0)
    final_pose = PlanarPose(0.3, 0, 0.8)
    mode.set_slider_initial_pose(initial_pose)
    mode.set_slider_final_pose(final_pose)

    mode.formulate_convex_relaxation()
    Xs = mode.get_Xs()
    result = MosekSolver().Solve(mode.relaxed_prog)  # type: ignore
    X_sols = [evaluate_np_expressions_array(X, result) for X in Xs]

    for X_val in X_sols:
        # Should be symmetric
        assert np.allclose(X_val - X_val.T, 0)


def test_face_contact_optimal_control_cost(plan_config: PlanarPlanConfig) -> None:
    plan_config.contact_config.slider_rot_velocity_constraint = 0.2
    plan_config.contact_config.slider_velocity_constraint = 0.05
    plan_config.use_band_sparsity = True
    plan_config.num_knot_points_contact = 4

    contact_location = PolytopeContactLocation(ContactLocation.FACE, 3)
    initial_pose = PlanarPose(0.3, 0.2, 1.0)
    final_pose = PlanarPose(0, 0, 0)

    plan_config.start_and_goal = PlanarPushingStartAndGoal(initial_pose, final_pose)

    mode = FaceContactMode.create_from_plan_spec(
        contact_location,
        plan_config,
    )
    mode.set_slider_initial_pose(initial_pose)
    mode.set_slider_final_pose(final_pose, hard_constraint=False)

    mode.formulate_convex_relaxation()
    solver = MosekSolver()
    result = solver.Solve(mode.relaxed_prog)  # type: ignore
    assert result.is_success()

    vars = mode.variables.eval_result(result)
    traj = PlanarPushingTrajectory(mode.config, [vars])

    if DEBUG:
        vars = mode.variables.eval_result(result)
        traj = PlanarPushingTrajectory(mode.config, [vars])

        visualize_planar_pushing_trajectory(
            traj, visualize_knot_points=True, save=True, filename="debug_file"
        )
        # (num_knot_points, 2): first col cosines, second col sines
        rs = np.vstack([R_WB[:, 0] for R_WB in traj.path_knot_points[0].R_WBs])  # type: ignore
        plot_cos_sine_trajs(rs)


def test_face_contact_euclidean_distance_cost(plan_config: PlanarPlanConfig) -> None:
    plan_config.use_band_sparsity = True
    plan_config.num_knot_points_contact = 4

    # Do not expect to get tight solutions with this, this is just to test the code
    plan_config.contact_config.cost.angular_arc_length = 0.1
    plan_config.contact_config.cost.linear_arc_length = 0.1
    plan_config.contact_config.cost.keypoint_arc_length = 0.1

    contact_location = PolytopeContactLocation(ContactLocation.FACE, 3)
    initial_pose = PlanarPose(0.3, 0.2, 1.0)
    final_pose = PlanarPose(0, 0, 0)

    plan_config.start_and_goal = PlanarPushingStartAndGoal(initial_pose, final_pose)

    mode = FaceContactMode.create_from_plan_spec(
        contact_location,
        plan_config,
    )
    mode.set_slider_initial_pose(initial_pose)
    mode.set_slider_final_pose(final_pose, hard_constraint=False)

    mode.formulate_convex_relaxation()
    solver = MosekSolver()
    result = solver.Solve(mode.relaxed_prog)  # type: ignore
    assert result.is_success()

    vars = mode.variables.eval_result(result)
    traj = PlanarPushingTrajectory(mode.config, [vars])

    if DEBUG:
        vars = mode.variables.eval_result(result)
        traj = PlanarPushingTrajectory(mode.config, [vars])

        visualize_planar_pushing_trajectory(
            traj, visualize_knot_points=True, save=True, filename="debug_file"
        )
        # (num_knot_points, 2): first col cosines, second col sines
        rs = np.vstack([R_WB[:, 0] for R_WB in traj.path_knot_points[0].R_WBs])  # type: ignore
        plot_cos_sine_trajs(rs)
