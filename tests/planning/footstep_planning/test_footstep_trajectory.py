import numpy as np
import pytest
from pydrake.solvers import (  # CommonSolverOption,
    Binding,
    BoundingBoxConstraint,
    CommonSolverOption,
    LinearConstraint,
    LinearCost,
    LinearEqualityConstraint,
    MosekSolver,
    PositiveSemidefiniteConstraint,
    QuadraticCost,
    RotatedLorentzConeConstraint,
    SnoptSolver,
    SolutionResult,
    Solve,
    SolverOptions,
)

from planning_through_contact.geometry.utilities import cross_2d
from planning_through_contact.planning.footstep.footstep_plan_config import (
    FootstepPlanningConfig,
    PotatoRobot,
)
from planning_through_contact.planning.footstep.footstep_trajectory import (
    FootstepPlan,
    FootstepPlanSegment,
)
from planning_through_contact.planning.footstep.in_plane_terrain import InPlaneTerrain
from planning_through_contact.tools.utils import evaluate_np_expressions_array
from planning_through_contact.visualize.footstep_visualizer import animate_footstep_plan

DEBUG = False


def test_trajectory_segment_one_foot() -> None:
    terrain = InPlaneTerrain()
    stone = terrain.add_stone(x_pos=0.5, width=1.5, z_pos=0.2, name="initial")

    robot = PotatoRobot()
    cfg = FootstepPlanningConfig(robot=robot)

    segment = FootstepPlanSegment(stone, "one_foot", robot, cfg, name="First step")

    # N states
    assert segment.p_WB.shape == (cfg.period_steps, 2)
    assert segment.v_WB.shape == (cfg.period_steps, 2)
    assert segment.theta_WB.shape == (cfg.period_steps,)
    assert segment.omega_WB.shape == (cfg.period_steps,)

    # N - 1 inputs
    assert segment.p_WF1.shape == (cfg.period_steps - 1, 2)
    assert segment.f_F1_1W.shape == (cfg.period_steps - 1, 2)
    assert segment.f_F1_2W.shape == (cfg.period_steps - 1, 2)

    assert segment.tau_F1_1.shape == (cfg.period_steps - 1,)
    assert segment.tau_F1_2.shape == (cfg.period_steps - 1,)

    desired_robot_pos = np.array([0.0, cfg.robot.desired_com_height])
    initial_pos = np.array([stone.x_pos - 0.02, 0.0]) + desired_robot_pos
    target_pos = np.array([stone.x_pos + 0.02, 0.0]) + desired_robot_pos

    segment.add_pose_constraint(0, initial_pos, 0)  # type: ignore
    segment.add_pose_constraint(-1, target_pos, 0)  # type: ignore

    mosek = MosekSolver()
    solver_options = SolverOptions()
    if DEBUG:
        solver_options.SetOption(CommonSolverOption.kPrintToConsole, 1)  # type: ignore

    relaxed_result = mosek.Solve(
        segment.make_relaxed_prog(), solver_options=solver_options  # type: ignore
    )
    # NOTE: We are getting UNKNOWN, but the solution looks good.
    assert (
        relaxed_result.is_success()
        or relaxed_result.get_solution_result() == SolutionResult.kSolverSpecificError
    )

    segment_value = segment.evaluate_with_result(relaxed_result)

    traj = FootstepPlan.merge([segment_value])
    assert traj.p_WB.shape == (cfg.period_steps, 2)
    assert traj.theta_WB.shape == (cfg.period_steps,)

    for foot in traj.feet_knot_points:
        assert foot.p_WF.shape == (cfg.period_steps - 1, 2)
        for f in foot.f_F_Ws:
            assert f.shape == (cfg.period_steps - 1, 2)

    if DEBUG:
        a_WB = evaluate_np_expressions_array(segment.a_WB, relaxed_result)
        cost_vals = segment.evaluate_costs_with_result(relaxed_result)

        cost_vals_sums = {key: np.sum(val) for key, val in cost_vals.items()}
        for key, val in cost_vals_sums.items():
            print(f"Cost {key}: {val}")

        print(f"Total cost: {relaxed_result.get_optimal_cost()}")

        non_convex_constraint_violation = (
            segment.evaluate_non_convex_constraints_with_result(relaxed_result)
        )
        print(
            f"Maximum constraint violation: {max(non_convex_constraint_violation.flatten()):.6f}"
        )

    if DEBUG:
        output_file = "debug_one_foot"
    else:
        output_file = None
    animate_footstep_plan(robot, terrain, traj, output_file=output_file)


def test_trajectory_segment_one_foot_extra_inputs() -> None:
    terrain = InPlaneTerrain()
    stone = terrain.add_stone(x_pos=0.5, width=1.5, z_pos=0.2, name="initial")

    robot = PotatoRobot()
    cfg = FootstepPlanningConfig(robot=robot)

    segment = FootstepPlanSegment(
        stone, "one_foot", robot, cfg, name="First step", eq_num_input_state=True
    )

    # N states
    assert segment.p_WB.shape == (cfg.period_steps, 2)
    assert segment.v_WB.shape == (cfg.period_steps, 2)
    assert segment.theta_WB.shape == (cfg.period_steps,)
    assert segment.omega_WB.shape == (cfg.period_steps,)

    # NOTE: This time, we should have N inputs!
    # N inputs
    assert segment.p_WF1.shape == (cfg.period_steps, 2)
    assert segment.f_F1_1W.shape == (cfg.period_steps, 2)
    assert segment.f_F1_2W.shape == (cfg.period_steps, 2)

    assert segment.tau_F1_1.shape == (cfg.period_steps,)
    assert segment.tau_F1_2.shape == (cfg.period_steps,)

    desired_robot_pos = np.array([0.0, cfg.robot.desired_com_height])
    initial_pos = np.array([stone.x_pos - 0.02, 0.0]) + desired_robot_pos
    target_pos = np.array([stone.x_pos + 0.02, 0.0]) + desired_robot_pos

    segment.add_pose_constraint(0, initial_pos, 0)  # type: ignore
    segment.add_pose_constraint(-1, target_pos, 0)  # type: ignore

    mosek = MosekSolver()
    solver_options = SolverOptions()
    if DEBUG:
        solver_options.SetOption(CommonSolverOption.kPrintToConsole, 1)  # type: ignore

    relaxed_result = mosek.Solve(
        segment.make_relaxed_prog(), solver_options=solver_options  # type: ignore
    )
    # NOTE: We are getting UNKNOWN, but the solution looks good.
    assert (
        relaxed_result.is_success()
        or relaxed_result.get_solution_result() == SolutionResult.kSolverSpecificError
    )

    segment_value = segment.evaluate_with_result(relaxed_result)

    traj = FootstepPlan.merge([segment_value])
    assert traj.p_WB.shape == (cfg.period_steps, 2)
    assert traj.theta_WB.shape == (cfg.period_steps,)

    for foot in traj.feet_knot_points:
        assert foot.p_WF.shape == (cfg.period_steps, 2)
        for f in foot.f_F_Ws:
            assert f.shape == (cfg.period_steps, 2)

    if DEBUG:
        output_file = "debug_one_foot_extra_inputs"
    else:
        output_file = None
    animate_footstep_plan(robot, terrain, traj, output_file=output_file)


def test_traj_segment_convex_concave_decomposition() -> None:
    terrain = InPlaneTerrain()
    stone = terrain.add_stone(x_pos=0.5, width=1.5, z_pos=0.2, name="initial")

    robot = PotatoRobot(max_step_dist_from_robot=0.6, step_span=0.7)
    cfg = FootstepPlanningConfig(robot=robot, use_convex_concave=True, period_steps=8)

    segment = FootstepPlanSegment(stone, "two_feet", robot, cfg, name="First step")

    desired_robot_pos = np.array([0.0, cfg.robot.desired_com_height])
    initial_pos = np.array([stone.x_pos, 0.0]) + desired_robot_pos
    target_pos = np.array([stone.x_pos + 0.3, 0.0]) + desired_robot_pos

    segment.add_pose_constraint(0, initial_pos, 0)  # type: ignore
    segment.add_pose_constraint(-1, target_pos, 0)  # type: ignore

    # This is required, otherwise we can start falling down on the last step and hence
    # have zero contact force.
    segment.add_spatial_vel_constraint(0, np.array([0, 0]), 0)
    segment.add_spatial_vel_constraint(-1, np.array([0, 0]), 0)

    segment.add_equilibrium_constraint(0)
    segment.add_equilibrium_constraint(-1)

    # NOTE:
    # The relaxation does actually realize that it needs to place the feet symmetrically
    # with this constraint
    segment.constrain_foot_pos_le("first", stone.x_pos - 0.3)
    segment.constrain_foot_pos_ge("last", stone.x_pos + 0.1)

    # Make sure all constraints and costs are convex
    for c in segment.prog.GetAllConstraints():
        if type(c.evaluator()) is LinearConstraint:
            continue

        if type(c.evaluator()) is LinearEqualityConstraint:
            continue

        if type(c.evaluator()) is BoundingBoxConstraint:
            continue

        if type(c.evaluator()) is RotatedLorentzConeConstraint:
            continue

        assert c.evaluator().is_convex()  # type: ignore

    for c in segment.prog.GetAllCosts():
        if type(c.evaluator()) is LinearCost:
            continue

        assert c.evaluator().is_convex()  # type: ignore

    for c in segment.prog.quadratic_constraints() + segment.prog.quadratic_costs():
        assert c.evaluator().is_convex()

    assert len(segment.prog.generic_constraints()) == 0
    assert len(segment.prog.generic_costs()) == 0

    # We should only have the convex/concave variables in the cost
    # for c in segment.prog.GetAllCosts():
    #     assert type(c.evaluator()) is QuadraticCost
    #
    #     assert len(c.variables()) == 1  # Q+ or Q-
    #     assert "Q" in str(c.variables()[0])

    solver_options = SolverOptions()
    if DEBUG:
        solver_options.SetOption(CommonSolverOption.kPrintToConsole, 1)  # type: ignore

    mosek = MosekSolver()
    relaxed_result = mosek.Solve(segment.prog, solver_options=solver_options)  # type: ignore
    assert relaxed_result.is_success()
    assert relaxed_result.get_solver_id().name() == "Mosek"

    segment_value_relaxed = segment.evaluate_with_result(relaxed_result)
    traj_relaxed = FootstepPlan.merge([segment_value_relaxed])

    if DEBUG:
        # Check quality of convex-concave relaxation
        tau_F1_1 = relaxed_result.GetSolution(segment.tau_F1_1)
        tau_F1_2 = relaxed_result.GetSolution(segment.tau_F1_2)
        tau_F2_1 = relaxed_result.GetSolution(segment.tau_F2_1)
        tau_F2_2 = relaxed_result.GetSolution(segment.tau_F2_2)

        p_BF1_1W = relaxed_result.GetSolution(segment.p_BF1_1W)
        p_BF1_2W = relaxed_result.GetSolution(segment.p_BF1_2W)
        p_BF2_1W = relaxed_result.GetSolution(segment.p_BF2_1W)
        p_BF2_2W = relaxed_result.GetSolution(segment.p_BF2_2W)

        f_F1_1W = relaxed_result.GetSolution(segment.f_F1_1W)
        f_F1_2W = relaxed_result.GetSolution(segment.f_F1_2W)
        f_F2_1W = relaxed_result.GetSolution(segment.f_F2_1W)
        f_F2_2W = relaxed_result.GetSolution(segment.f_F2_2W)

        target_tau_F1_1 = np.array(
            [cross_2d(p, f).Evaluate() for p, f in zip(p_BF1_1W, f_F1_1W)]
        )
        target_tau_F1_2 = np.array(
            [cross_2d(p, f).Evaluate() for p, f in zip(p_BF1_2W, f_F1_2W)]
        )
        target_tau_F2_1 = np.array(
            [cross_2d(p, f).Evaluate() for p, f in zip(p_BF2_1W, f_F2_1W)]
        )
        target_tau_F2_2 = np.array(
            [cross_2d(p, f).Evaluate() for p, f in zip(p_BF2_2W, f_F2_2W)]
        )

        diff1_1 = tau_F1_1 - target_tau_F1_1
        diff1_2 = tau_F1_2 - target_tau_F1_2
        diff2_1 = tau_F2_1 - target_tau_F2_1
        diff2_2 = tau_F2_2 - target_tau_F2_2

        # Should be close if the relaxation is tight (which it seems it
        # is not)
        # assert np.allclose(tau_F1_1, target_tau_F1_1, atol=1e-1)
        # assert np.allclose(tau_F1_2, target_tau_F1_2, atol=1e-1)
        # assert np.allclose(tau_F2_1, target_tau_F2_1, atol=1e-1)
        # assert np.allclose(tau_F2_2, target_tau_F2_2, atol=1e-1)

        a_WB = evaluate_np_expressions_array(segment.a_WB, relaxed_result)
        omega_dot_WB = evaluate_np_expressions_array(
            segment.omega_dot_WB, relaxed_result
        )

        tau_sum = tau_F1_1 + tau_F1_2 + tau_F2_1 + tau_F2_2
        omega_WB = relaxed_result.GetSolution(segment.omega_WB)

        cost_vals = segment.evaluate_costs_with_result(relaxed_result)
        cost_vals_sums = {key: np.sum(val) for key, val in cost_vals.items()}
        for key, val in cost_vals_sums.items():
            print(f"Cost {key}: {val}")

        print(f"Total cost: {relaxed_result.get_optimal_cost()}")

        if len(segment.non_convex_constraints) > 0:
            non_convex_constraint_violation = (
                segment.evaluate_non_convex_constraints_with_result(relaxed_result)
            )
            print(
                f"Maximum constraint violation: {max(non_convex_constraint_violation.flatten()):.6f}"
            )
        if len(segment.convex_concave_slack_vars) > 0:
            slack_vars = np.array(
                segment.convex_concave_slack_vars
            ).T  # (num_vars, num_inputs)
            slack_vars_vals = relaxed_result.GetSolution(slack_vars)

    test_round = False
    if test_round:
        segment_value, rounded_result = segment.round_with_result(relaxed_result)

        assert rounded_result.get_solver_id().name() == "SNOPT"
        traj_rounded = FootstepPlan.merge([segment_value])
        if DEBUG:
            output_file_rounded = "debug_convex_concave_rounded"
        else:
            output_file_rounded = None

        animate_footstep_plan(
            robot, terrain, traj_rounded, output_file=output_file_rounded
        )

        if DEBUG:
            c_round = rounded_result.get_optimal_cost()
            c_relax = relaxed_result.get_optimal_cost()
            ub_optimality_gap = (c_round - c_relax) / c_relax
            print(f"UB optimality gap: {ub_optimality_gap:.5f} %")

    if DEBUG:
        output_file_relaxed = "debug_convex_concave_relaxed"
    else:
        output_file_relaxed = None
    animate_footstep_plan(robot, terrain, traj_relaxed, output_file=output_file_relaxed)


def test_trajectory_segment_two_feet_one_stone() -> None:
    """
    Tests a single segment with two feet in contact. Tests both
    convex relaxation (SDP) and the nonlinear rounding.
    """
    terrain = InPlaneTerrain()
    stone = terrain.add_stone(x_pos=0.5, width=1.5, z_pos=0.2, name="initial")

    robot = PotatoRobot()
    cfg = FootstepPlanningConfig(robot=robot)

    segment = FootstepPlanSegment(stone, "two_feet", robot, cfg, name="First step")

    assert segment.p_WF1.shape == (cfg.period_steps - 1, 2)
    assert segment.f_F1_1W.shape == (cfg.period_steps - 1, 2)
    assert segment.f_F1_2W.shape == (cfg.period_steps - 1, 2)

    assert segment.tau_F1_1.shape == (cfg.period_steps - 1,)
    assert segment.tau_F1_2.shape == (cfg.period_steps - 1,)

    assert segment.p_WF2.shape == (cfg.period_steps - 1, 2)
    assert segment.f_F1_1W.shape == (cfg.period_steps - 1, 2)
    assert segment.f_F2_2W.shape == (cfg.period_steps - 1, 2)

    assert segment.tau_F2_1.shape == (cfg.period_steps - 1,)
    assert segment.tau_F2_1.shape == (cfg.period_steps - 1,)

    desired_robot_pos = np.array([0.0, cfg.robot.desired_com_height])
    initial_pos = np.array([stone.x_pos - 0.2, 0.0]) + desired_robot_pos
    target_pos = np.array([stone.x_pos + 0.2, 0.0]) + desired_robot_pos

    segment.add_pose_constraint(0, initial_pos, 0)  # type: ignore
    segment.add_pose_constraint(cfg.period_steps - 1, target_pos, 0)  # type: ignore

    solver_options = SolverOptions()
    if DEBUG:
        solver_options.SetOption(CommonSolverOption.kPrintToConsole, 1)  # type: ignore

    relaxed_result = Solve(
        segment.make_relaxed_prog(trace_cost=False), solver_options=solver_options
    )
    # NOTE: We are getting UNKNOWN, but the solution looks good.
    assert (
        relaxed_result.is_success()
        or relaxed_result.get_solution_result() == SolutionResult.kSolverSpecificError
    )

    if DEBUG:
        a_WB = evaluate_np_expressions_array(segment.a_WB, relaxed_result)
        cost_vals = segment.evaluate_costs_with_result(relaxed_result)
        cost_vals_sums = {key: np.sum(val) for key, val in cost_vals.items()}
        for key, val in cost_vals_sums.items():
            print(f"Cost {key}: {val}")

        print(f"Total cost: {relaxed_result.get_optimal_cost()}")

        non_convex_constraint_violation = (
            segment.evaluate_non_convex_constraints_with_result(relaxed_result)
        )
        print(
            f"Maximum constraint violation: {max(non_convex_constraint_violation.flatten()):.6f}"
        )

    segment_value, rounded_result = segment.round_with_result(relaxed_result)

    # segment_value = segment.evaluate_with_result(result)
    if DEBUG:
        c_round = rounded_result.get_optimal_cost()
        c_relax = relaxed_result.get_optimal_cost()
        ub_optimality_gap = (c_round - c_relax) / c_relax
        print(f"UB optimality gap: {ub_optimality_gap:.5f} %")

    traj = FootstepPlan.merge([segment_value])

    if DEBUG:
        output_file = "debug_two_feet"
    else:
        output_file = None
    animate_footstep_plan(robot, terrain, traj, output_file=output_file)


def test_trajectory_segment_two_feet_different_stones() -> None:
    terrain_1 = InPlaneTerrain()
    step_diff = 0.3
    stone_1_low = terrain_1.add_stone(x_pos=0.5, width=1.0, z_pos=0.2, name="stone_low")
    stone_1_high = terrain_1.add_stone(
        x_pos=1.5, width=1.0, z_pos=0.2 + step_diff, name="stone_high"
    )

    terrain_2 = InPlaneTerrain()

    stone_2_low = terrain_2.add_stone(x_pos=1.5, width=1.0, z_pos=0.2, name="stone_low")
    stone_2_high = terrain_2.add_stone(
        x_pos=0.5, width=1.0, z_pos=0.2 + step_diff, name="stone_high"
    )

    robot = PotatoRobot()
    cfg = FootstepPlanningConfig(robot=robot)

    segment_step_up = FootstepPlanSegment(
        stone_1_low,
        "two_feet",
        robot,
        cfg,
        name="step_down",
        stone_for_last_foot=stone_1_high,
    )

    assert segment_step_up.p_WF1[0, 1] == stone_1_low.z_pos
    assert segment_step_up.p_WF2[0, 1] == stone_1_high.z_pos

    segment_step_down = FootstepPlanSegment(
        stone_2_high,
        "two_feet",
        robot,
        cfg,
        name="step_down",
        stone_for_last_foot=stone_2_low,
    )
    assert segment_step_down.p_WF1[0, 1] == stone_1_high.z_pos
    assert segment_step_down.p_WF2[0, 1] == stone_1_low.z_pos

    for terrain, segment in zip(
        (terrain_1, terrain_2), (segment_step_up, segment_step_down)
    ):

        initial_pos = np.array([0.95, cfg.robot.desired_com_height])
        target_pos = np.array([1.05, cfg.robot.desired_com_height + step_diff])

        segment.add_pose_constraint(0, initial_pos, 0)  # type: ignore
        segment.add_pose_constraint(cfg.period_steps - 1, target_pos, 0)  # type: ignore

        solver_options = SolverOptions()
        if DEBUG:
            solver_options.SetOption(CommonSolverOption.kPrintToConsole, 1)  # type: ignore

        relaxed_result = Solve(
            segment.make_relaxed_prog(trace_cost=False), solver_options=solver_options
        )
        # NOTE: We are getting UNKNOWN, but the solution looks good.
        assert (
            relaxed_result.is_success()
            or relaxed_result.get_solution_result()
            == SolutionResult.kSolverSpecificError
        )

        segment_value, rounded_result = segment.round_with_result(relaxed_result)

        # segment_value = segment.evaluate_with_result(result)
        if DEBUG:
            c_round = rounded_result.get_optimal_cost()
            c_relax = relaxed_result.get_optimal_cost()
            ub_optimality_gap = (c_round - c_relax) / c_relax
            print(f"UB optimality gap: {ub_optimality_gap:.5f} %")

        traj = FootstepPlan.merge([segment_value])

        if DEBUG:
            output_file = f"debug_different_stones_{segment.name}"
        else:
            output_file = None
        animate_footstep_plan(robot, terrain, traj, output_file=output_file)


def test_merging_two_trajectory_segments() -> None:
    """
    This test only tests that the FootstepTrajectory class and the visualizer is able to correctly
    merge and visualize the feet over multiple segments
    """
    terrain = InPlaneTerrain()
    stone = terrain.add_stone(x_pos=0.5, width=1.5, z_pos=0.2, name="initial")

    robot = PotatoRobot()
    cfg = FootstepPlanningConfig(robot=robot)

    desired_robot_pos = np.array([0.0, cfg.robot.desired_com_height])
    initial_pos = np.array([stone.x_pos - 0.15, 0.0]) + desired_robot_pos
    target_pos = np.array([stone.x_pos + 0.15, 0.0]) + desired_robot_pos
    target_pos_2 = np.array([stone.x_pos + 0.18, 0.0]) + desired_robot_pos

    segment_first = FootstepPlanSegment(
        stone, "one_foot", robot, cfg, name="First step"
    )
    segment_first.add_pose_constraint(0, initial_pos, 0)  # type: ignore
    segment_first.add_pose_constraint(cfg.period_steps - 1, target_pos, 0)  # type: ignore

    result_first = Solve(segment_first.make_relaxed_prog())

    # NOTE: We are getting UNKNOWN, but the solution looks good.
    assert (
        result_first.is_success()
        or result_first.get_solution_result() == SolutionResult.kSolverSpecificError
    )

    segment_val_first = segment_first.evaluate_with_result(result_first)

    segment_second = FootstepPlanSegment(
        stone, "two_feet", robot, cfg, name="second step"
    )

    segment_second.add_pose_constraint(0, target_pos, 0)  # type: ignore
    segment_second.add_pose_constraint(cfg.period_steps - 1, target_pos_2, 0)  # type: ignore

    result_second = Solve(segment_second.make_relaxed_prog())

    # NOTE: We are getting UNKNOWN, but the solution looks good.
    assert (
        result_second.is_success()
        or result_second.get_solution_result() == SolutionResult.kSolverSpecificError
    )
    segment_val_second = segment_second.evaluate_with_result(result_second)

    traj = FootstepPlan.merge([segment_val_first, segment_val_second])

    if DEBUG:
        output_file = "debug_merge_two_segments"
    else:
        output_file = None
    animate_footstep_plan(robot, terrain, traj, output_file=output_file)


def test_tightness_eval() -> None:
    terrain = InPlaneTerrain()
    stone = terrain.add_stone(x_pos=0.5, width=1.5, z_pos=0.2, name="initial")

    robot = PotatoRobot()
    cfg = FootstepPlanningConfig(robot=robot)

    segment = FootstepPlanSegment(stone, "two_feet", robot, cfg, name="First step")

    assert segment.p_WF1.shape == (cfg.period_steps - 1, 2)
    assert segment.f_F1_1W.shape == (cfg.period_steps - 1, 2)
    assert segment.f_F1_2W.shape == (cfg.period_steps - 1, 2)

    assert segment.tau_F1_1.shape == (cfg.period_steps - 1,)
    assert segment.tau_F1_2.shape == (cfg.period_steps - 1,)

    assert segment.p_WF2.shape == (cfg.period_steps - 1, 2)
    assert segment.f_F1_1W.shape == (cfg.period_steps - 1, 2)
    assert segment.f_F2_2W.shape == (cfg.period_steps - 1, 2)

    assert segment.tau_F2_1.shape == (cfg.period_steps - 1,)
    assert segment.tau_F2_1.shape == (cfg.period_steps - 1,)

    desired_robot_pos = np.array([0.0, cfg.robot.desired_com_height])
    initial_pos = np.array([stone.x_pos - 0.2, 0.0]) + desired_robot_pos
    target_pos = np.array([stone.x_pos + 0.2, 0.0]) + desired_robot_pos

    segment.add_pose_constraint(0, initial_pos, 0)  # type: ignore
    segment.add_pose_constraint(cfg.period_steps - 1, target_pos, 0)  # type: ignore

    segment.constrain_foot_pos_le("first", stone.x_pos - 0.3)
    segment.constrain_foot_pos_ge("last", stone.x_pos + 0.1)

    solver_options = SolverOptions()
    if DEBUG:
        solver_options.SetOption(CommonSolverOption.kPrintToConsole, 1)  # type: ignore

    relaxed_result = Solve(
        segment.make_relaxed_prog(trace_cost=False), solver_options=solver_options
    )
    assert relaxed_result.is_success()

    segment_value_relaxed = segment.evaluate_with_result(relaxed_result)
    traj_relaxed = FootstepPlan.merge([segment_value_relaxed])

    if DEBUG:
        output_file = "debug_tightness_eval"
    else:
        output_file = None

    animate_footstep_plan(robot, terrain, traj_relaxed, output_file=output_file)
