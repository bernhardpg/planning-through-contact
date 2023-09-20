import numpy as np
import pytest
from pydrake.solvers import Solve

from planning_through_contact.geometry.planar.face_contact import FaceContactMode
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.planar.planar_pushing_path import (
    assemble_progs_from_contact_modes,
)
from planning_through_contact.geometry.planar.trajectory_builder import (
    PlanarTrajectoryBuilder,
)
from planning_through_contact.planning.planar.planar_plan_config import (
    PlanarSolverParams,
)
from planning_through_contact.planning.planar.planar_pushing_planner import (
    PlanarPushingPlanner,
)
from planning_through_contact.visualize.analysis import plot_cos_sine_trajs
from planning_through_contact.visualize.planar import (
    visualize_planar_pushing_trajectory,
)
from tests.geometry.planar.fixtures import (
    box_geometry,
    dynamics_config,
    face_contact_mode,
    plan_config,
    planner,
    rigid_body_box,
    t_pusher,
)
from tests.geometry.planar.tools import assert_initial_and_final_poses

DEBUG = False


def test_rounding_one_mode(face_contact_mode: FaceContactMode) -> None:
    initial_pose = PlanarPose(0, 0, 0)
    final_pose = PlanarPose(0.3, 0, 0.8)
    face_contact_mode.set_slider_initial_pose(initial_pose)
    face_contact_mode.set_slider_final_pose(final_pose)

    face_contact_mode.formulate_convex_relaxation()

    assert face_contact_mode.relaxed_prog is not None

    relaxed_result = Solve(face_contact_mode.relaxed_prog)
    assert relaxed_result.is_success()

    prog = assemble_progs_from_contact_modes([face_contact_mode])
    initial_guess = relaxed_result.GetSolution(
        face_contact_mode.relaxed_prog.decision_variables()[: prog.num_vars()]
    )

    prog.SetInitialGuess(prog.decision_variables(), initial_guess)
    result = Solve(prog)

    vars = face_contact_mode.variables.eval_result(result)
    traj = PlanarTrajectoryBuilder([vars]).get_trajectory(interpolate=False)

    assert_initial_and_final_poses(traj, initial_pose, None, final_pose, None)

    if DEBUG:
        visualize_planar_pushing_trajectory(
            traj, face_contact_mode.config.slider_geometry, 0.01
        )
        # (num_knot_points, 2): first col cosines, second col sines
        rs = np.vstack([R_WB[:, 0] for R_WB in traj.R_WB])
        plot_cos_sine_trajs(rs)


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
def test_path_with_teleportation(planner: PlanarPushingPlanner) -> None:
    solver_params = PlanarSolverParams(print_solver_output=DEBUG)
    result = planner._solve(solver_params)
    assert result.is_success()
    path = planner.get_solution_path(result)

    # We should always have one more vertex than edge in a solution
    assert len(path.pairs) == len(path.edges) + 1

    prog = path._construct_nonlinear_program()

    expected_num_vars = sum([p.mode.prog.num_vars() for p in path.pairs])
    assert prog.num_vars() == expected_num_vars

    expected_num_constraints = sum(
        [len(p.mode.prog.GetAllConstraints()) for p in path.pairs]
    ) + sum([len(edge.GetConstraints()) for edge in path.edges])
    assert len(prog.GetAllConstraints()) == expected_num_constraints


@pytest.mark.parametrize(
    "planner",
    [
        {
            "partial": True,
            "allow_teleportation": True,
            "use_redundant_dynamic_constraints": False,
            "boundary_conds": {
                "finger_initial_pose": PlanarPose(x=0, y=-0.5, theta=0.0),
                "finger_target_pose": PlanarPose(x=-0.3, y=0, theta=0.0),
                "box_initial_pose": PlanarPose(x=0, y=0, theta=0.0),
                "box_target_pose": PlanarPose(x=-0.2, y=-0.2, theta=0.4),
            },
        },
        {
            "partial": True,
            "allow_teleportation": True,
            "use_redundant_dynamic_constraints": False,
            "boundary_conds": {
                "finger_initial_pose": PlanarPose(x=0, y=-0.5, theta=0.0),
                "finger_target_pose": PlanarPose(x=0, y=-0.5, theta=0.0),
                "box_initial_pose": PlanarPose(x=0, y=0, theta=0.0),
                "box_target_pose": PlanarPose(x=-0.2, y=-0.2, theta=1.2),
            },
        },
        {
            "partial": True,
            "avoid_object": True,
            "allow_teleportation": False,
            "use_redundant_dynamic_constraints": False,
            "boundary_conds": {
                "finger_initial_pose": PlanarPose(x=0, y=-0.5, theta=0.0),
                "finger_target_pose": PlanarPose(x=0, y=-0.5, theta=0.0),
                "box_initial_pose": PlanarPose(x=0, y=0, theta=0.0),
                "box_target_pose": PlanarPose(x=-0.2, y=-0.2, theta=1.2),
            },
        },
        # Very slow, so commented out. Should pass!
        # {
        #     "partial": False,
        #     "avoid_object": True,
        #     "allow_teleportation": False,
        #     "use_redundant_dynamic_constraints": False,
        #     "boundary_conds": {
        #         "finger_initial_pose": PlanarPose(x=0.2, y=0.2, theta=0.0),
        #         "finger_target_pose": PlanarPose(x=0.2, y=0.2, theta=0.0),
        #         "box_initial_pose": PlanarPose(x=0.2, y=0, theta=1.2),
        #         "box_target_pose": PlanarPose(x=-0.2, y=-0.2, theta=2.1),
        #     },
        # },
    ],
    indirect=["planner"],
    ids=[
        "teleport1",
        "teleport2",
        "normal1",
        # "normal2",
    ],
)
def test_path_rounding(planner: PlanarPushingPlanner) -> None:
    solver_params = PlanarSolverParams(print_solver_output=DEBUG)
    result = planner._solve(solver_params)
    assert result.is_success()

    path = planner.get_solution_path(result)
    rounded_result = path._do_nonlinear_rounding(print_output=DEBUG, measure_time=DEBUG)
    assert rounded_result.is_success()

    traj = PlanarTrajectoryBuilder(path.get_vars()).get_trajectory(interpolate=False)
    traj_rounded = PlanarTrajectoryBuilder(path.get_rounded_vars()).get_trajectory(
        interpolate=False
    )

    if DEBUG:
        # visualize_planar_pushing_trajectory(traj, planner.slider.geometry)
        visualize_planar_pushing_trajectory(traj_rounded, planner.slider.geometry)
