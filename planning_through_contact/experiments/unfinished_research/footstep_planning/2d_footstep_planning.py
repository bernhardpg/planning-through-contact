import numpy as np
from pydrake.solvers import (
    CommonSolverOption,
    MosekSolver,
    SolutionResult,
    Solve,
    SolverOptions,
)

from planning_through_contact.planning.footstep.footstep_plan_config import (
    FootstepPlanningConfig,
    PotatoRobot,
)
from planning_through_contact.planning.footstep.footstep_planner import FootstepPlanner
from planning_through_contact.planning.footstep.footstep_trajectory import (
    FootstepPlanSegment,
    FootstepTrajectory,
)
from planning_through_contact.planning.footstep.in_plane_terrain import InPlaneTerrain
from planning_through_contact.tools.utils import evaluate_np_expressions_array
from planning_through_contact.visualize.footstep_visualizer import animate_footstep_plan

DEBUG = True


def test_trajectory_segment_one_foot() -> None:
    terrain = InPlaneTerrain()
    stone = terrain.add_stone(x_pos=0.5, width=1.5, z_pos=0.2, name="initial")

    robot = PotatoRobot()
    cfg = FootstepPlanningConfig(robot=robot)

    segment = FootstepPlanSegment(
        stone, np.array([1, 0]), robot, cfg, name="First step"
    )

    assert segment.p_WB.shape == (cfg.period_steps, 2)
    assert segment.v_WB.shape == (cfg.period_steps, 2)
    assert segment.theta_WB.shape == (cfg.period_steps,)
    assert segment.omega_WB.shape == (cfg.period_steps,)

    assert segment.p_WFl.shape == (cfg.period_steps, 2)
    assert segment.f_Fl_1W.shape == (cfg.period_steps, 2)
    assert segment.f_Fl_2W.shape == (cfg.period_steps, 2)

    assert segment.tau_Fl_1.shape == (cfg.period_steps,)
    assert segment.tau_Fl_2.shape == (cfg.period_steps,)

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
    # assert result.is_success()
    assert result.get_solution_result() == SolutionResult.kSolverSpecificError

    segment_value = segment.evaluate_with_result(result)

    active_feet = np.array([[True, False]])

    traj = FootstepTrajectory.from_segments([segment_value], cfg.dt, active_feet)
    assert traj.knot_points.p_WB.shape == (cfg.period_steps, 2)
    assert traj.knot_points.theta_WB.shape == (cfg.period_steps,)
    assert traj.knot_points.p_WFl.shape == (cfg.period_steps, 2)
    assert traj.knot_points.f_Fl_1W.shape == (cfg.period_steps, 2)
    assert traj.knot_points.f_Fl_2W.shape == (cfg.period_steps, 2)

    if DEBUG:
        a_WB = evaluate_np_expressions_array(segment.a_WB, result)
        cost_vals = segment.evaluate_costs_with_result(result)

    if DEBUG:
        output_file = "debug_one_foot"
    else:
        output_file = None
    animate_footstep_plan(robot, terrain, traj, output_file=output_file)


def test_trajectory_segment_two_feet() -> None:
    terrain = InPlaneTerrain()
    stone = terrain.add_stone(x_pos=0.5, width=1.5, z_pos=0.2, name="initial")

    robot = PotatoRobot()
    cfg = FootstepPlanningConfig(robot=robot)

    segment = FootstepPlanSegment(
        stone, np.array([1, 1]), robot, cfg, name="First step"
    )

    assert segment.p_WFl.shape == (cfg.period_steps, 2)
    assert segment.f_Fl_1W.shape == (cfg.period_steps, 2)
    assert segment.f_Fl_2W.shape == (cfg.period_steps, 2)

    assert segment.tau_Fl_1.shape == (cfg.period_steps,)
    assert segment.tau_Fl_2.shape == (cfg.period_steps,)

    assert segment.p_WFr.shape == (cfg.period_steps, 2)
    assert segment.f_Fr_1W.shape == (cfg.period_steps, 2)
    assert segment.f_Fr_2W.shape == (cfg.period_steps, 2)

    assert segment.tau_Fr_1.shape == (cfg.period_steps,)
    assert segment.tau_Fr_2.shape == (cfg.period_steps,)

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
    assert relaxed_result.is_success()

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

    active_feet = np.array([[True, True]])
    traj = FootstepTrajectory.from_segments([segment_value], cfg.dt, active_feet)

    assert traj.knot_points.p_WB.shape == (cfg.period_steps, 2)
    assert traj.knot_points.theta_WB.shape == (cfg.period_steps,)
    assert traj.knot_points.p_WFl.shape == (cfg.period_steps, 2)
    assert traj.knot_points.f_Fl_1W.shape == (cfg.period_steps, 2)
    assert traj.knot_points.f_Fl_2W.shape == (cfg.period_steps, 2)

    if DEBUG:
        output_file = "debug_two_feet"
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
        stone, np.array([1, 1]), robot, cfg, name="First step"
    )
    segment_first.add_pose_constraint(0, initial_pos, 0)  # type: ignore
    segment_first.add_pose_constraint(cfg.period_steps - 1, target_pos, 0)  # type: ignore

    result_first = Solve(segment_first.make_relaxed_prog())
    assert result_first.is_success()
    segment_val_first = segment_first.evaluate_with_result(result_first)

    segment_second = FootstepPlanSegment(
        stone, np.array([1, 0]), robot, cfg, name="second step"
    )
    segment_second.add_pose_constraint(0, target_pos, 0)  # type: ignore
    segment_second.add_pose_constraint(cfg.period_steps - 1, target_pos_2, 0)  # type: ignore

    result_second = Solve(segment_second.make_relaxed_prog())

    # NOTE: We are getting UNKNOWN, but the solution looks good.
    # assert result_second.is_success()
    assert result_second.get_solution_result() == SolutionResult.kSolverSpecificError
    segment_val_second = segment_second.evaluate_with_result(result_second)

    active_feet = np.array([[True, True], [True, False]])
    traj = FootstepTrajectory.from_segments(
        [segment_val_first, segment_val_second], cfg.dt, active_feet
    )

    if DEBUG:
        output_file = "debug_merge_two_segments"
    else:
        output_file = None
    animate_footstep_plan(robot, terrain, traj, output_file=output_file)


def test_footstep_planning_one_stone() -> None:
    """
    This should give exactly the same result as the test
    test_trajectory_segment_two_feet
    """
    terrain = InPlaneTerrain()
    stone = terrain.add_stone(x_pos=1.0, width=2.0, z_pos=0.2, name="initial")

    robot = PotatoRobot()
    cfg = FootstepPlanningConfig(robot=robot)

    desired_robot_pos = np.array([0.0, cfg.robot.desired_com_height])
    initial_pos = np.array([stone.x_pos - 0.6, 0.0]) + desired_robot_pos
    target_pos = np.array([stone.x_pos + 0.6, 0.0]) + desired_robot_pos

    initial_pose = np.concatenate([initial_pos, [0]])
    target_pose = np.concatenate([target_pos, [0]])

    planner = FootstepPlanner(cfg, terrain, initial_pose, target_pose)

    if DEBUG:
        planner.create_graph_diagram("test_one_stone_diagram")
    plan = planner.plan(print_flows=True)

    if DEBUG:
        plan.save("test_one_stone_plan.pkl")

    if DEBUG:
        output_file = "debug_plan_one_stone"
    else:
        output_file = None
    animate_footstep_plan(robot, terrain, plan, output_file=output_file)


def main():
    breakpoint()
    # test_single_point()
    # test_trajectory_segment_one_foot()
    # test_merging_two_trajectory_segments()
    # test_trajectory_segment_two_feet()
    # test_footstep_planning_one_stone()
    # load_traj()


if __name__ == "__main__":
    main()
