import os

import numpy as np
import pytest
from pydrake.solvers import CommonSolverOption, MosekSolver, SnoptSolver, SolverOptions

from planning_through_contact.experiments.utils import (
    get_default_plan_config,
    get_sugar_box,
)
from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    ContactLocation,
    PolytopeContactLocation,
)
from planning_through_contact.geometry.planar.face_contact import FaceContactMode
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.planar.planar_pushing_path import (
    assemble_progs_from_contact_modes,
)
from planning_through_contact.geometry.planar.planar_pushing_trajectory import (
    PlanarPushingTrajectory,
)
from planning_through_contact.planning.planar.planar_plan_config import (
    PlanarPushingStartAndGoal,
    PlanarSolverParams,
)
from planning_through_contact.planning.planar.planar_pushing_planner import (
    PlanarPushingPlanner,
)
from planning_through_contact.visualize.analysis import (
    analyze_plan,
    plot_cos_sine_trajs,
)
from planning_through_contact.visualize.planar_pushing import (
    make_traj_figure,
    visualize_planar_pushing_trajectory,
)
from tests.geometry.planar.tools import assert_initial_and_final_poses

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"

DEBUG = False


def test_rounding_one_mode() -> None:
    config = get_default_plan_config()
    initial_pose = PlanarPose(0, 0, 0)
    final_pose = PlanarPose(0.2, 0.05, 0.4)
    config.start_and_goal = PlanarPushingStartAndGoal(initial_pose, final_pose)
    contact_location = PolytopeContactLocation(ContactLocation.FACE, 3)
    mode = FaceContactMode.create_from_plan_spec(
        contact_location,
        config,
    )

    mode.set_slider_initial_pose(initial_pose)
    mode.set_slider_final_pose(final_pose)

    mode.formulate_convex_relaxation()

    assert mode.relaxed_prog is not None

    relaxed_result = MosekSolver().Solve(mode.relaxed_prog)  # type: ignore
    assert relaxed_result.is_success()
    relaxed_vars = mode.variables.eval_result(relaxed_result)

    if DEBUG:
        relaxed_vars = mode.variables.eval_result(relaxed_result)
        relaxed_traj = PlanarPushingTrajectory(mode.config, [relaxed_vars])
        visualize_planar_pushing_trajectory(
            relaxed_traj, visualize_knot_points=True, save=True, filename="debug_file"
        )
        make_traj_figure(relaxed_traj, filename="debug_file")

    prog = assemble_progs_from_contact_modes([mode])
    initial_guess = relaxed_result.GetSolution(prog.decision_variables())

    solver_options = SolverOptions()

    if DEBUG:
        solver_options.SetOption(CommonSolverOption.kPrintToConsole, 1)  # type: ignore

    snopt = SnoptSolver()
    solver_options.SetOption(snopt.solver_id(), "Print file", "debug_solver_log.txt")
    result = snopt.Solve(prog, initial_guess=initial_guess, solver_options=solver_options)  # type: ignore

    assert result.is_success()

    vars = mode.variables.eval_result(result)

    traj = PlanarPushingTrajectory(mode.config, [vars])
    assert_initial_and_final_poses(traj, initial_pose, None, final_pose, None)

    if DEBUG:
        vars = mode.variables.eval_result(result)
        traj = PlanarPushingTrajectory(mode.config, [vars])

        visualize_planar_pushing_trajectory(
            traj, visualize_knot_points=True, save=True, filename="debug_file_rounded"
        )
        make_traj_figure(traj, filename="debug_file_rounded")
        # (num_knot_points, 2): first col cosines, second col sines
        rs_relaxed = np.vstack([R_WB[:, 0] for R_WB in relaxed_vars.R_WBs])
        plot_cos_sine_trajs(rs_relaxed, filename="debug_cos_sin")
        rs = np.vstack([R_WB[:, 0] for R_WB in vars.R_WBs])
        plot_cos_sine_trajs(rs, filename="debug_cos_sin_rounded")


@pytest.mark.skipif(
    IN_GITHUB_ACTIONS == True,
    reason="Too slow",
)
@pytest.mark.parametrize(
    "planner",
    [
        (
            {
                "partial": True,
                "allow_teleportation": True,
                "penalize_mode_transition": False,
                "boundary_conds": {
                    "finger_initial_pose": PlanarPose(x=0, y=-0.4, theta=0.0),
                    "finger_target_pose": PlanarPose(x=-0.3, y=0, theta=0.0),
                    "box_initial_pose": PlanarPose(x=0, y=0, theta=0.0),
                    "box_target_pose": PlanarPose(x=-0.2, y=-0.2, theta=0.4),
                },
            }
        ),
    ],
    indirect=["planner"],
)
def test_path_construction_with_teleportation(planner: PlanarPushingPlanner) -> None:
    solver_params = PlanarSolverParams(print_solver_output=DEBUG)
    result = planner._solve(solver_params)
    assert result.is_success()
    path = planner.get_solution_paths(result, solver_params)[0]

    # We should always have one more vertex than edge in a solution
    assert len(path.pairs) == len(path.edges) + 1

    prog = path._construct_nonlinear_program()

    expected_num_vars = sum([p.mode.prog.num_vars() for p in path.pairs])  # type: ignore
    assert prog.num_vars() == expected_num_vars

    expected_num_constraints = sum([len(p.mode.prog.GetAllConstraints()) for p in path.pairs]) + sum(  # type: ignore
        [len(edge.GetConstraints()) for edge in path.edges]
    )
    assert len(prog.GetAllConstraints()) == expected_num_constraints


@pytest.mark.parametrize(
    "plan_spec",
    [
        PlanarPushingStartAndGoal(
            PlanarPose(x=0, y=0, theta=0.0),
            PlanarPose(x=0.0, y=0, theta=0.0),
            PlanarPose(x=-0.2, y=-0.2, theta=0.0),
            PlanarPose(x=-0.2, y=-0.2, theta=0.0),
        ),
        PlanarPushingStartAndGoal(
            PlanarPose(x=0.2, y=0.2, theta=1.0),
            PlanarPose(x=0, y=0, theta=0.0),
            PlanarPose(x=-0.2, y=-0.2, theta=0.0),
            PlanarPose(x=-0.2, y=-0.2, theta=0.0),
        ),
        PlanarPushingStartAndGoal(
            PlanarPose(x=0, y=0, theta=-2.0),
            PlanarPose(x=-0.2, y=0.2, theta=0.0),
            PlanarPose(x=0.2, y=0.0, theta=0.0),
            PlanarPose(x=0.2, y=0.0, theta=0.0),
        ),
    ],
    ids=[1, 2, 3],
)
def test_path_rounding(plan_spec: PlanarPushingStartAndGoal) -> None:
    get_sugar_box()
    config = get_default_plan_config(pusher_radius=0.035)
    planner = PlanarPushingPlanner(config)
    solver_params = PlanarSolverParams(
        measure_solve_time=DEBUG,
        print_solver_output=DEBUG,
        save_solver_output=DEBUG,
        assert_result=False,
    )
    planner.config.start_and_goal = plan_spec
    planner.formulate_problem()

    path = planner.plan_path(solver_params)
    traj_relaxed = path.to_traj()

    if DEBUG:
        planner.create_graph_diagram(filename="debug_graph")
        visualize_planar_pushing_trajectory(
            traj_relaxed,
            visualize_knot_points=True,
            save=True,
            filename="debug_file_relaxed",
        )
        make_traj_figure(traj_relaxed, filename="debug_file_relaxed")
        analyze_plan(path, filename="debug_analysis_relaxed", rounded=False)

    assert path.rounded_result is not None
    assert path.rounded_result.is_success()

    traj_rounded = PlanarPushingTrajectory(planner.config, path.get_rounded_vars())
    traj_rounded = path.to_traj(rounded=True)

    if DEBUG:
        visualize_planar_pushing_trajectory(
            traj_rounded,
            visualize_knot_points=True,
            save=True,
            filename="debug_file_rounded",
        )
        make_traj_figure(traj_rounded, filename="debug_file_rounded")
        analyze_plan(path, filename="debug_analysis_rounded", rounded=True)
