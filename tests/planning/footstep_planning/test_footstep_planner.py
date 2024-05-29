import numpy as np
import pytest
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
from planning_through_contact.planning.footstep.footstep_planner import FootstepPlanner
from planning_through_contact.planning.footstep.footstep_trajectory import (
    FootstepPlanSegmentProgram,
    get_X_from_semidefinite_relaxation,
)
from planning_through_contact.planning.footstep.in_plane_terrain import InPlaneTerrain
from planning_through_contact.tools.utils import evaluate_np_expressions_array
from planning_through_contact.visualize.footstep_visualizer import animate_footstep_plan

DEBUG = True


def test_footstep_planning_one_stone() -> None:
    terrain = InPlaneTerrain()
    stone = terrain.add_stone(x_pos=1.0, width=1.0, z_pos=0.2, name="initial")

    robot = PotatoRobot()
    cfg = FootstepPlanningConfig(robot=robot)

    desired_robot_pos = np.array([0.0, cfg.robot.desired_com_height])
    desired_displacement = 0.2
    initial_pos = (
        np.array([stone.x_pos - desired_displacement, stone.z_pos]) + desired_robot_pos
    )
    target_pos = (
        np.array([stone.x_pos + desired_displacement, stone.z_pos]) + desired_robot_pos
    )

    initial_pose = np.concatenate([initial_pos, [0]])
    target_pose = np.concatenate([target_pos, [0]])

    planner = FootstepPlanner(
        cfg,
        terrain,
        initial_pose,
        target_pose,
        initial_stone_name=stone.name,
        target_stone_name=stone.name,
    )

    if DEBUG:
        planner.create_graph_diagram("test_one_stone_diagram")
    plan = planner.plan(print_flows=DEBUG, print_solver_output=DEBUG, print_debug=DEBUG)

    if DEBUG:
        plan.save("test_one_stone_plan.pkl")

    if DEBUG:
        output_file = "debug_plan_one_stone"
    else:
        output_file = None
    animate_footstep_plan(robot, terrain, plan, output_file=output_file)


def test_footstep_planning_two_stones() -> None:
    terrain = InPlaneTerrain()
    initial_stone = terrain.add_stone(x_pos=0.25, width=0.5, z_pos=0.2, name="initial")
    target_stone = terrain.add_stone(x_pos=0.75, width=0.5, z_pos=0.5, name="target")

    robot = PotatoRobot()
    cfg = FootstepPlanningConfig(robot=robot)

    desired_robot_pos = np.array([0.0, cfg.robot.desired_com_height])
    initial_pos = initial_stone.com + desired_robot_pos
    target_pos = target_stone.com + desired_robot_pos

    # no body rotation
    initial_pose = np.concatenate([initial_pos, [0]])
    target_pose = np.concatenate([target_pos, [0]])

    planner = FootstepPlanner(
        cfg,
        terrain,
        initial_pose,
        target_pose,
        initial_stone_name=initial_stone.name,
        target_stone_name=target_stone.name,
    )

    if DEBUG:
        planner.create_graph_diagram("test_two_stones_diagram")

    plan = planner.plan(print_flows=True, print_solver_output=DEBUG)

    if DEBUG:
        output_file = "debug_plan_two_stones"
    else:
        output_file = None
    animate_footstep_plan(robot, terrain, plan, output_file=output_file)


def test_footstep_planning_one_long_stone_lp_approx() -> None:
    terrain = InPlaneTerrain()
    stone = terrain.add_stone(x_pos=1.0, width=2.0, z_pos=0.2, name="initial")

    robot = PotatoRobot()
    cfg = FootstepPlanningConfig(robot=robot, use_lp_approx=True)

    desired_robot_pos = np.array([0.0, cfg.robot.desired_com_height + stone.z_pos])
    initial_pos = np.array([stone.x_pos - 0.6, 0.0]) + desired_robot_pos
    target_pos = np.array([stone.x_pos + 0.6, 0.0]) + desired_robot_pos

    initial_pose = np.concatenate([initial_pos, [0]])
    target_pose = np.concatenate([target_pos, [0]])

    planner = FootstepPlanner(
        cfg,
        terrain,
        initial_pose,
        target_pose,
        initial_stone_name=stone.name,
        target_stone_name=stone.name,
    )

    planner.plan(print_flows=True, print_solver_output=DEBUG)

    if DEBUG:
        planner.save_analysis("test_one_stone_lp_approx")


def test_footstep_planning_many_stones_lp_approx() -> None:
    terrain = InPlaneTerrain()
    initial_stone = terrain.add_stone(x_pos=0.25, width=0.5, z_pos=0.2, name="initial")
    _ = terrain.add_stone(x_pos=1.0, width=1.0, z_pos=0.5, name="stone_2")
    _ = terrain.add_stone(x_pos=2.0, width=1.0, z_pos=0.7, name="stone_3")
    _ = terrain.add_stone(x_pos=2.75, width=0.5, z_pos=0.9, name="stone_4")
    _ = terrain.add_stone(x_pos=3.25, width=0.5, z_pos=0.7, name="stone_5")
    target_stone = terrain.add_stone(x_pos=3.75, width=0.5, z_pos=0.5, name="target")

    robot = PotatoRobot()
    cfg = FootstepPlanningConfig(robot=robot, use_lp_approx=True, period=0.3)

    desired_robot_pos = np.array([0.0, cfg.robot.desired_com_height])
    initial_pos = initial_stone.com + desired_robot_pos
    target_pos = target_stone.com + desired_robot_pos

    # no body rotation
    initial_pose = np.concatenate([initial_pos, [0]])
    target_pose = np.concatenate([target_pos, [0]])

    planner = FootstepPlanner(
        cfg,
        terrain,
        initial_pose,
        target_pose,
        initial_stone_name=initial_stone.name,
        target_stone_name=target_stone.name,
    )

    if DEBUG:
        planner.create_graph_diagram("test_many_stones_lp_approx_diagram")

    plan = planner.plan(print_flows=True, print_solver_output=DEBUG)

    if DEBUG:
        output_file = "debug_plan_many_stones_lp_approx_rounded"
    else:
        output_file = None
    animate_footstep_plan(robot, terrain, plan, output_file=output_file)

    if DEBUG:
        output_file = "debug_plan_many_stones_lp_approx_relaxation"
    else:
        output_file = None
    animate_footstep_plan(
        robot, terrain, planner.get_relaxed_plan(), output_file=output_file
    )


# Unfinished! This is meant to be a WIP on a cutting-plane algorithm on
# lp approximation
@pytest.mark.skip
def test_semidefinite_relaxation_lp_approximation() -> None:
    terrain = InPlaneTerrain()
    stone = terrain.add_stone(x_pos=0.5, width=1.5, z_pos=0.2, name="initial")

    robot = PotatoRobot()
    cfg = FootstepPlanningConfig(robot=robot)

    segment = FootstepPlanSegmentProgram(
        stone, "two_feet", robot, cfg, name="First step"
    )

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

    # Make relaxation
    relaxed_prog = segment.make_relaxed_prog()

    assert len(relaxed_prog.positive_semidefinite_constraints()) == 1
    X = get_X_from_semidefinite_relaxation(relaxed_prog)

    sdp_constraint = relaxed_prog.positive_semidefinite_constraints()[0]

    relaxed_prog.RemoveConstraint(sdp_constraint)  # type: ignore

    N = X.shape[0]
    for i in range(N):
        X_i = X[i, i]
        relaxed_prog.AddLinearConstraint(X_i >= 0)

    # np.random.seed(0)
    # for i in range(5):
    #     v = np.random.rand(N, 1)
    #     v = v / np.linalg.norm(v)
    #     V = v @ v.T
    #     relaxed_prog.AddLinearConstraint(np.sum(X * V) >= 0)

    relaxed_result = Solve(relaxed_prog, solver_options=solver_options)
    assert relaxed_result.is_success()

    relaxed_prog.AddConstraint(sdp_constraint.evaluator(), sdp_constraint.variables())
    X_val = relaxed_result.GetSolution(X)

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
        output_file = "debug_lp_approximation"
    else:
        output_file = None
    animate_footstep_plan(robot, terrain, traj, output_file=output_file)


def test_make_segments_per_terrain() -> None:
    terrain = InPlaneTerrain()
    stone = terrain.add_stone(x_pos=1.0, width=2.0, z_pos=0.2, name="initial")

    step_span = 0.5

    robot = PotatoRobot(step_span=step_span)
    cfg = FootstepPlanningConfig(robot=robot)

    desired_robot_pos = np.array([0.0, cfg.robot.desired_com_height])
    initial_pos = np.array([stone.x_pos - 0.6, 0.0]) + desired_robot_pos
    target_pos = np.array([stone.x_pos + 0.6, 0.0]) + desired_robot_pos

    initial_pose = np.concatenate([initial_pos, [0]])
    target_pose = np.concatenate([target_pos, [0]])

    planner = FootstepPlanner(
        cfg,
        terrain,
        initial_pose,
        target_pose,
        initial_stone_name="initial",
        target_stone_name="initial",
    )
    segments = planner._make_segments_for_terrain()
    DEBUG = True

    if DEBUG:
        planner.create_graph_diagram("test_make_segments_diagram")

    # we should have one segment per stone
    assert len(segments) == len(terrain.stepping_stones)

    def _expected_num_steps(_stone):
        return (
            planner._calc_num_steps_required_per_stone(_stone.width, step_span) * 2 - 1
        )

    # We add 2 here, as the "start_stance" and "final_stance" are added to the first and last segments,
    # which in this case is the same
    assert len(segments[0]) == _expected_num_steps(stone) + 2

    # Add another stone and try again

    stone_2 = terrain.add_stone(x_pos=3.0, width=2.0, z_pos=0.2, name="stone_2")
    planner = FootstepPlanner(
        cfg,
        terrain,
        initial_pose,
        target_pose,
        initial_stone_name="initial",
        target_stone_name="stone_2",
    )
    segments = planner._make_segments_for_terrain()

    if DEBUG:
        planner.create_graph_diagram("test_make_segments_diagram_2")

    assert len(segments) == len(terrain.stepping_stones)

    # Make sure we have the correct number of steps given the step length
    # We add 1 for both here, as the "start_stance" and "final_stance" are added to the first and last segments,
    assert len(segments[0]) == _expected_num_steps(stone) + 1
    assert len(segments[1]) == _expected_num_steps(stone) + 1
