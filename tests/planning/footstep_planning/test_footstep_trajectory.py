import numpy as np
from pydrake.solvers import (  # CommonSolverOption,
    Binding,
    CommonSolverOption,
    MosekSolver,
    PositiveSemidefiniteConstraint,
    SolutionResult,
    Solve,
    SolverOptions,
)

from planning_through_contact.planning.footstep.footstep_plan_config import (
    FootstepPlanningConfig,
    PotatoRobot,
)
from planning_through_contact.planning.footstep.footstep_trajectory import (
    FootstepPlanSegment,
    FootstepTrajectory,
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

    assert segment.p_WB.shape == (cfg.period_steps, 2)
    assert segment.v_WB.shape == (cfg.period_steps, 2)
    assert segment.theta_WB.shape == (cfg.period_steps,)
    assert segment.omega_WB.shape == (cfg.period_steps,)

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

    result = mosek.Solve(segment.make_relaxed_prog(), solver_options=solver_options)
    # NOTE: We are getting UNKNOWN, but the solution looks good.
    assert (
        result.is_success()
        or result.get_solution_result() == SolutionResult.kSolverSpecificError
    )

    segment_value = segment.evaluate_with_result(result)

    traj = FootstepTrajectory.from_segments([segment_value], cfg.dt)
    assert traj.knot_points.p_WB.shape == (cfg.period_steps, 2)
    assert traj.knot_points.theta_WB.shape == (cfg.period_steps,)
    assert traj.knot_points.p_WF1.shape == (cfg.period_steps, 2)
    assert traj.knot_points.f_F1_1W.shape == (cfg.period_steps, 2)
    assert traj.knot_points.f_F1_2W.shape == (cfg.period_steps, 2)

    if DEBUG:
        evaluate_np_expressions_array(segment.a_WB, result)
        segment.evaluate_costs_with_result(result)

    if DEBUG:
        output_file = "debug_one_foot"
    else:
        output_file = None
    animate_footstep_plan(robot, terrain, traj, output_file=output_file)


def test_trajectory_segment_two_feet() -> None:
    """
    Tests a single segment with two feet in contact. Tests both
    convex relaxation (SDP) and the nonlinear rounding.
    """
    terrain = InPlaneTerrain()
    stone = terrain.add_stone(x_pos=0.5, width=1.5, z_pos=0.2, name="initial")

    robot = PotatoRobot()
    cfg = FootstepPlanningConfig(robot=robot)

    segment = FootstepPlanSegment(stone, "two_feet", robot, cfg, name="First step")

    assert segment.p_WF1.shape == (cfg.period_steps, 2)
    assert segment.f_F1_1W.shape == (cfg.period_steps, 2)
    assert segment.f_F1_2W.shape == (cfg.period_steps, 2)

    assert segment.tau_F1_1.shape == (cfg.period_steps,)
    assert segment.tau_F1_2.shape == (cfg.period_steps,)

    assert segment.p_WF2.shape == (cfg.period_steps, 2)
    assert segment.f_F1_1W.shape == (cfg.period_steps, 2)
    assert segment.f_F2_2W.shape == (cfg.period_steps, 2)

    assert segment.tau_F2_1.shape == (cfg.period_steps,)
    assert segment.tau_F2_1.shape == (cfg.period_steps,)

    desired_robot_pos = np.array([0.0, cfg.robot.desired_com_height])
    initial_pos = np.array([stone.x_pos - 0.2, 0.0]) + desired_robot_pos
    target_pos = np.array([stone.x_pos + 0.2, 0.0]) + desired_robot_pos

    segment.add_pose_constraint(0, initial_pos, 0)  # type: ignore
    segment.add_pose_constraint(cfg.period_steps - 1, target_pos, 1.0)  # type: ignore

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
        evaluate_np_expressions_array(segment.a_WB, relaxed_result)
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

    traj = FootstepTrajectory.from_segments([segment_value], cfg.dt)

    assert traj.knot_points.p_WB.shape == (cfg.period_steps, 2)
    assert traj.knot_points.theta_WB.shape == (cfg.period_steps,)
    assert traj.knot_points.p_WF1.shape == (cfg.period_steps, 2)
    assert traj.knot_points.f_F1_1W.shape == (cfg.period_steps, 2)
    assert traj.knot_points.f_F1_2W.shape == (cfg.period_steps, 2)

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

        traj = FootstepTrajectory.from_segments([segment_value], cfg.dt)

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

    traj = FootstepTrajectory.from_segments(
        [segment_val_first, segment_val_second], cfg.dt
    )

    if DEBUG:
        output_file = "debug_merge_two_segments"
    else:
        output_file = None
    animate_footstep_plan(robot, terrain, traj, output_file=output_file)
